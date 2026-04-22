import typing, requests, time, io, json
from PIL import Image
from constructs.roi import ROI
from constructs.classification import Classification, LabelType
import base64
from utils.helper import print_green, print_yellow, print_red


# Keep WorkClient logs plain to avoid excessive terminal coloring.

class WorkClient(object):

    def __init__(self, gs_socket: str, cs_socket : str):

        # Specify port to listen on and endpoint
        self.gs_url = "http://" + gs_socket + "/"
        self.attribute_endp = "api/v1/targets/all" # Getting target attributes
        self.work_endp = "api/v1/assignment/work" # Getting work
        self.adlc_endp = "api/v1/gcp_target_sighting/assignment" # Sending ADLC output

        # Specify cloud server ports
        self.cs_url = "http://" + cs_socket + "/" 
        self.upload_img_endp = "api/images"
        self.tent_img_endp = "api/images/tent"
        self.mannequin_img_endp = "api/images/mannequin"

        # Specify our username to interact w/ gs
        self.client_header = {"id": 1, "username": "adlc",
                        "address": "", "userType": "ADLC"}
        self.auth_headers = {"Username": "adlc"}
        self.http_timeout_seconds = 5

    def _decode_base64_image(self, image_b64: str) -> typing.Optional[Image.Image]:
        if not image_b64:
            print_yellow("[work_client] Empty base64 payload; cannot decode image")
            return None

        # Support either raw base64 or data URLs (data:image/jpeg;base64,...)
        payload = image_b64.split(",", 1)[1] if "," in image_b64 else image_b64
        try:
            image_bytes = base64.b64decode(payload)
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            print_red(f"[work_client] Failed to decode base64 image: {e}")
            return None

    def _parse_label(self, raw_label: typing.Any) -> LabelType:
        if raw_label is None:
            return LabelType.UNKNOWN

        # Accept enum-like strings ("TENT") or raw enum values (0/1/2)
        if isinstance(raw_label, str):
            key = raw_label.strip().upper()
            return LabelType.__members__.get(key, LabelType.UNKNOWN)

        try:
            return LabelType(int(raw_label))
        except Exception:
            return LabelType.UNKNOWN


    def _parse_candidate_image(
        self, response: requests.Response
    ) -> tuple[typing.Optional[dict], typing.Optional[ROI], typing.Optional[Classification]]:
        if response.status_code == 204:
            print("[work_client] Candidate endpoint returned 204 (no content yet)")
            return None, None, None

        if response.status_code != 200:
            print_red(f"[work_client] Failed to get candidate image (status={response.status_code})")
            return None, None, None

        try:
            candidate = response.json()
        except Exception as e:
            print_red(f"[work_client] Invalid candidate JSON response: {e}")
            return None, None, None

        bbox = candidate.get("bbox") or []
        score = float(candidate.get("score", 0.0))
        label = self._parse_label(candidate.get("label"))

        source = candidate.get("source") or {}
        source_b64 = source.get("base64_image")
        assignment = source.get("assignment")
        full_image = self._decode_base64_image(source_b64)

        if full_image is None or len(bbox) != 4:
            print_yellow("[work_client] Candidate missing valid source image or 4-value bbox")
            return assignment, None, None
        
        width, height = full_image.size
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
        except Exception:
            print_red(f"[work_client] Invalid bbox format: {bbox}")
            return assignment, None, None

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            print_yellow(f"[work_client] Degenerate bbox after clipping: {[x1, y1, x2, y2]}")
            return assignment, None, None

        roi_image = full_image.crop((x1, y1, x2, y2))
        roi = ROI(roi=roi_image, top_left=(x1, y1), bottom_right=(x2, y2))
        classification = Classification(label=label, number_conf=score)

        return assignment, roi, classification

    def get_target_attributes(self) -> typing.Dict[str, list]:
        """
        Gets the attributes (color, shape, etc.) of targets before mission
        starts

        """
        while True:
            response = requests.get(self.gs_url + self.attribute_endp)
            status_code, attrs = response.status_code, dict(response.json())
            if status_code != 200 or not attrs:
                print_yellow(f"[work_client] Waiting for target attributes (status={status_code})")
                time.sleep(2)
                continue
            else:
                attrs_formatted = {}
                for id, target in attrs.items():
                    desc = [s.upper() for s in target.values()]
                    attrs_formatted[id] = desc
                    print(f"[work_client] Target {id}: {desc[1]} {desc[0]} with {desc[3]} {desc[2]}")
                return attrs_formatted

    def get_image_assignment(self) -> typing.Tuple[typing.Dict[str, str], typing.Dict[str, str]]:
        """
        Before we try and access the actual image, we first access its metadata
        in order to filter out bad images, and get the URL of the image to access
        it if we decide we want to. We also set self.assignment to the resçponse of
        the request, since we need it for sending the ADLC output.

        """
        while True:
            response = requests.post(
                self.gs_url + self.work_endp, headers=self.auth_headers
            )
            status_code = response.status_code
            print(f"[work_client] Polled work endpoint: {self.gs_url + self.work_endp} (status={status_code})")
            if status_code == 204:  # successful request, no content
                print("[work_client] No work assignment available yet")
                time.sleep(2)
                continue
            elif status_code == 200:  # successful request, received image metadata
                assignment = response.json()
                meta = dict(assignment)
                data = {
                    "id": meta["id"],
                    "endpoint": meta["image"]["imageUrl"],
                    "timestamp": meta["image"]["timestamp"],
                    "telemetry": meta["image"]["telemetry"],
                    "imgMode": meta["image"]["imgMode"],
                }
                print(f"[work_client] Received assignment {data['id']} (endpoint={data['endpoint']})")
                return assignment, data
            else:
                print_red(f"[work_client] Work request failed (status={status_code})")
                break

    def get_image(self, img_endpoint: str) -> Image.Image:
        """
        Sends a request for the image given the endpoint, taken from metadata.

        """
        response = requests.get(self.gs_url + img_endpoint)

        if response.status_code == 200:
            image = Image.open(io.BytesIO(response.content))
            print(f"[work_client] Received image from GS endpoint: {img_endpoint}")
            return image
        else:
            print_red(f"[work_client] Failed to fetch image from GS (status={response.status_code}, endpoint={img_endpoint})")
            return None
        
    def send_image(self, img: Image.Image, assignment: dict) -> requests.Response:
        if img is None:
            print_red("[work_client] Cannot send image to cloud: image is None")
            raise ValueError("img cannot be None")
        if assignment is None or "id" not in assignment:
            print_red("[work_client] Cannot send image to cloud: missing assignment or assignment id")
            raise ValueError("assignment with id is required")

        buffer = io.BytesIO()
        img_format = img.format if img.format else "PNG"
        img.save(buffer, format=img_format)
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        payload = {"base64_image": img_base64, "id": assignment["id"], "meta": None, "assignment": assignment}
        
        # Send request to cloud server
        response = requests.post(
            self.cs_url + self.upload_img_endp,
            json=payload,  # automatically sets Content-Type: application/json
            timeout=self.http_timeout_seconds,
        )

        if 200 <= response.status_code < 300:
            print(f"[work_client] Uploaded image to cloud (assignment_id={assignment['id']}, status={response.status_code})")
        else:
            print_red(f"[work_client] Cloud upload failed (assignment_id={assignment['id']}, status={response.status_code})")

        return response

    def get_tent_image(self) -> tuple[typing.Optional[dict], typing.Optional[ROI], typing.Optional[Classification]]:
        try:
            response = requests.get(
                self.cs_url + self.tent_img_endp,
                timeout=self.http_timeout_seconds,
            )
        except requests.RequestException as e:
            print_red(f"[work_client] Tent request failed or timed out: {e}")
            return None, None, None

        if response.status_code == 204:
            print_yellow("[work_client] No tent image available from cloud yet (204)")
        return self._parse_candidate_image(response)
    
    def get_mannequin_image(self) -> tuple[typing.Optional[dict], typing.Optional[ROI], typing.Optional[Classification]]:
        try:
            response = requests.get(
                self.cs_url + self.mannequin_img_endp,
                timeout=self.http_timeout_seconds,
            )
        except requests.RequestException as e:
            print_red(f"[work_client] Mannequin request failed or timed out: {e}")
            return None, None, None

        if response.status_code == 204:
            print_yellow("[work_client] No mannequin image available from cloud yet (204)")
        return self._parse_candidate_image(response)


    def send_adlc_output(self, assignment: dict, roi : ROI, classification : Classification) -> requests.Response:
        """
        Posts each ROI and its classifications.

        Args:
            assignment: assignment of the image
            roi: roi of the target
            classification: classification of the target
        
        Returns:
            The response from the server.
        """
        # shape, shape_conf = classification.shape
        # shape_color, shape_color_conf = classification.shape_color
        # alphanumeric, alphanumeric_conf = classification.alpha_num
        # alpha_color, alpha_color_conf = classification.alpha_color
        label, number_conf = classification.label
        # TODO: Implement angle if needed
        # angle = roi.orientation.angle()
        angle = 0.0
        x_coord, y_coord = roi.center
        width = roi.width
        height = roi.height
        data = {
            "creator": self.client_header,
            "assignment": assignment,

            "targetLabel": str(label), # tent, mannequin, none
            "targetConfidence": number_conf,
            
            "pixelx": x_coord,
            "pixely": y_coord,
            "radiansFromTop": angle,
            "offaxis": False,
            "width": width,
            "height": height,
            # 'targetId': targetId,
        }
        print(
            f"[work_client] Sending ADLC output to GS "
            f"(assignment_id={assignment.get('id')}, label={label}, conf={number_conf:.3f}, "
            f"center=({x_coord},{y_coord}), size=({width}x{height}))"
        )

        response = requests.post(
            f"{self.gs_url}{self.adlc_endp}/{assignment['id']}",
            headers=self.auth_headers,
            json=data,
            timeout=self.http_timeout_seconds,
        )
        if 200 <= response.status_code < 300:
            print(f"[work_client] ADLC output accepted by GS (status={response.status_code})")
        else:
            print_red(f"[work_client] ADLC output rejected by GS (status={response.status_code})")
            if response.text:
                print_red(f"[work_client] GS response body: {response.text}")
        return response

if __name__ == "__main__":
    wc = WorkClient("127.0.0.1:9000")
    wc.get_image("/api/v1/image/file/1726429003.jpeg")
