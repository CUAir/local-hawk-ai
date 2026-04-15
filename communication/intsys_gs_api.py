from http.server import BaseHTTPRequestHandler
import json
import os
from utils.helper import print_green, print_red, print_yellow

class MapCommandHandler(BaseHTTPRequestHandler):
    mapper = None
    
    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else '{}'
            
            data = json.loads(post_data)
            command = data.get('command', '')
            
            if command == 'start':
                self.mapper.mapping = True
                response = {"status": "success", "message": "Mapping started"}
                print_green("Mapping started!")
            elif command == 'stop':
                self.mapper.mapping = False
                response = {"status": "success", "message": "Mapping stopped"}
                print_green("Mapping stopped!")
            elif command == 'generate':
                print_yellow("Generating map...")
                self.mapper.generate_map()
                response = {"status": "success", "message": "Map generated"}
            else:
                response = {"status": "error", "message": f"Unknown command: {command}"}
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            return response
        except Exception as e:
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            error_response = {"status": "error", "message": str(e)}
            print_red(f"Error: {error_response}")
            self.wfile.write(json.dumps(error_response).encode())
            return error_response

    def do_DELETE(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else '{}'
            data = json.loads(post_data)
            
            image_id = data.get('image_id')
            if not image_id:
                raise ValueError("image_id is required for delete")
            
            image_path = os.path.join('images', f'{image_id}.jpg')
            if os.path.exists(image_path):
                os.remove(image_path)
                # Remove entry from CSV
                with open('images.csv', 'r') as f:
                    lines = f.readlines()
                with open('images.csv', 'w') as f:
                    for line in lines:
                        if not line.startswith(image_id + ','):
                            f.write(line)
                
                # Remove from processed images
                self.mapper.processed_images = [img for img in self.mapper.processed_images 
                                              if img.get('image_id') != image_id]
                
                response = {"status": "success", "message": f"Image {image_id} deleted"}
                print_green(f"Image {image_id} deleted")

                self.send_response(204)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                return response 
            else:
                self.send_response(404)
                response = {"status": "error", "message": f"Image {image_id} not found"}
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                return response
            
        except Exception as e:
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            error_response = {"status": "error", "message": str(e)}
            print_red(f"Error: {error_response}")
            self.wfile.write(json.dumps(error_response).encode())
            return error_response