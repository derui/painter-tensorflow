type t = 
    StartFileLoading of string
  | EndFileLoading of (string * int * int)
  | StartImageDragging
  | MoveImage of (int * int)
  | EndImageDragging

let to_string = function
  | StartFileLoading _ -> "start_file_loading"
  | EndFileLoading _ -> "end_file_loading"
  | StartImageDragging -> "end_image_dragging"
  | MoveImage _ -> "move_image"
  | EndImageDragging -> "end_image_dragging"

(* Actions for image dragging *)
let start_image_dragging () = StartImageDragging
let end_image_dragging () = EndImageDragging
let move_image x y = MoveImage (x, y)

(* Load image from file object *)
let load_file dispatch file =
  let open Bs_dom_wrapper in
  let name = File.name file in
  let reader = File_reader.create () in
  File_reader.setOnload reader (fun _ ->
      let result = File_reader.result reader in
      let img = Html.Image.create () in
      Html.Image.setOnload img (fun _ ->
          let width = Html.Image.width img
          and height = Html.Image.height img in
          dispatch (EndFileLoading (result, width, height))
        );
      Html.Image.setSrc img result;
    );
  reader |> File_reader.readAsDataURL file;
  dispatch (StartFileLoading name)
