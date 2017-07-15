(* define reducer and state type *)
type state = {
    file_name: string;
    choosed_image: string;
  }

(* reduce state from action *)
let reduce state = function
  | `StartFileLoading file -> {state with file_name = file}
  | `EndFileLoading result -> {state with choosed_image = result}
  | _ -> state
