type t = [
    `StartFileLoading of string
  | `EndFileLoading of string
  ]

let to_string = function
  | `ChangeFile _ -> "change_file"
  | `StartFileLoading _ -> "start_file_loading"
  | `EndFileLoading _ -> "end_file_loading"

type 'a dispatch = 'a -> ('a -> t) -> unit

let load_file (dispatch:'a dispatch) file =
  let open Dom_util in
  let name = File.name file in
  let reader = FileReader.create () in
  FileReader.set_onload reader (fun e ->
      let target = e |> Progress_event.get_target in
      let result = FileReader.get_result target in
      dispatch result (fun v -> `EndFileLoading v)
    );
  FileReader.read_as_data_url reader file;
  dispatch name (fun v -> `StartFileLoading v)
