open Cvat_data_annotation

(** Define a map from competitions to functions that construct Csv files in that
    format. *)
let csv_types =
  [ ("cuasc-25", Cuasc_25.construct_csv); ("suas-24", Suas_24.construct_csv) ]

type cli_arguments = {
  input_dir : string;
  output_fp : string;
  csv_constructor : string -> string -> unit;
}
(** Type to consolidate command line arguments*)

(** [parse_command_line ()] is a Result.t wrapping a collection of arguments
    that reveals the input directory path where CVAT [.json] files live
    (containing annotated data), the output filepath for the [.csv] constructed
    from these [.json]s, and a function to actually construct this csv.
    Otherwise, it's a Result.t wrapping an error-message string. *)
let parse_command_line (cli_args : string Array.t) :
    (cli_arguments, string) Result.t =
  let (usage_msg, argument_exists) : string * (int -> (string, string) Result.t)
      =
    "USAGE: dune exec bin/main.exe <CVAT `.json`s directory> <output `.csv` \
     dump filepath> <competition (e.g. one of: "
    ^ (csv_types |> List.map fst |> String.concat ", ")
    ^ ")>"
    |> fun usage_msg ->
    ( (* helpful error message for output *)
      usage_msg,
      (* Returns Ok(string) if the argument exists at [index], otherwise Error(usage_msg) *)
      fun (index : int) ->
        try Result.Ok cli_args.(index) with _ -> Result.error usage_msg )
  in
  (* define infix operator for propagating Result.t monad; improves readability *)
  let ( >>= ) = Result.bind in
  argument_exists 1 >>= fun maybe_input_dir ->
  argument_exists 2 >>= fun output_fp ->
  argument_exists 3 >>= fun competition_type ->
  (try
     if Sys.is_directory maybe_input_dir then Result.Ok maybe_input_dir
     else Result.error usage_msg
   with _ -> Result.error usage_msg)
  >>= fun input_dir ->
  match List.assoc_opt competition_type csv_types with
  | None -> Result.error usage_msg
  | Some func -> Result.Ok { input_dir; output_fp; csv_constructor = func }

let () =
  Sys.argv |> parse_command_line
  |> fun (result : (cli_arguments, string) Result.t) ->
  match result with
  | Result.Error msg -> failwith msg
  | Result.Ok { input_dir; output_fp; csv_constructor } ->
      csv_constructor input_dir output_fp
