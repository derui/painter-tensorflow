(* define reducer and state type *)
type state = {
  file: string;
}

(* reduce state from action *)
let reduce state = function
  | `ChangeFile file -> {file = file}
  | _ -> state
