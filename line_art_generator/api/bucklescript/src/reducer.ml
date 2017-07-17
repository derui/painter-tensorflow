(* define reducer and state type *)
let max_scaled_size = 256
let max_selector_size = 512

module Size = struct
  type t = {
      width: int;
      height: int;
    }

  let mul t v = {width = t.width * v;height = t.height * v}
  let div t v = {width = t.width / v;height = t.height / v}
end

(* type for image map *)
type image_map = {
    original_size: Size.t;
    scaled_size: Size.t;
    scaling_factor: float;
    selector_position: int * int;
    selector_size: Size.t;
  }

type state = {
    file_name: string;
    choosed_image: string;
    image_map: image_map option;
    dragging: bool;
  }

let empty = {
    file_name = "";
    choosed_image = "";
    image_map = None;
    dragging = false;
  }

let calc_scaling_factor original max_size =
  if original.Size.width >= original.Size.height then
    if original.Size.width > max_size then
      (float_of_int original.Size.width) /. (float_of_int max_size)
    else if original.Size.height > max_size then
      (float_of_int original.Size.height) /. (float_of_int max_size)
    else
      1.0
  else
    if original.Size.height > max_size then
      (float_of_int original.Size.height) /. (float_of_int max_size)
    else if original.Size.width > max_size then
      (float_of_int original.Size.width) /. (float_of_int max_size)
    else
      1.0


let scale_size original scaling_factor =
  let scale v = float_of_int v |> (fun v -> ceil (v /. scaling_factor))
                |> int_of_float
  in
  {Size.width = scale original.Size.width; height = scale original.Size.height}

let make_image_map width height = 
  let original_size = {Size.width; height} in 
  let selector_size = {Size.width = min width max_selector_size;
                       height = min height max_selector_size} in
  let scaling_factor = calc_scaling_factor original_size max_scaled_size in
  let scaled_size = scale_size original_size scaling_factor in 
  let selector_size = scale_size selector_size scaling_factor in
  {original_size;
   scaled_size;
   scaling_factor;
   selector_position = (0, 0);
   selector_size;
  }

let handle_end_file_loading state (result, width, height) =
  {state with choosed_image = result;
              image_map = Some (make_image_map width height)
  }

(* reduce state from action *)
let reduce state = function
  | Actions.StartFileLoading file -> {state with file_name = file}
  | Actions.EndFileLoading action -> handle_end_file_loading state action
  | Actions.StartImageDragging -> {state with dragging = true}
  | Actions.EndImageDragging -> {state with dragging = false}
  | Actions.MoveImage pos -> begin
      match state.image_map with
      | None -> state
      | Some image_map ->
         let size = Size.div image_map.selector_size 2 in
         let scaled_size = image_map.scaled_size in
         let x, y = pos in
         let x = x - size.Size.width
         and y = y - size.Size.height in
         let range min max v = Pervasives.min (Pervasives.max min v) max in
         {state with image_map = Some {image_map with selector_position = (
                                     Size.(range 0 (scaled_size.width - size.width * 2) x),
                                     Size.(range 0 (scaled_size.height - size.height * 2) y)
                                   )}
         }
    end
