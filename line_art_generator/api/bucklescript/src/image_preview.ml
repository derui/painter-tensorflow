(* Define function to update  *)

let update_canvas state canvas =
  let context = canvas |> Dom_util.Canvas_element.get_context "2d" in
  if state.Reducer.choosed_image = "" then ()
  else begin
      let module I = Dom_util.Image_element in
      let module C = Dom_util.Canvas_context in
      let img = I.create () in 
      I.set_onload img (fun _ ->
          context |> C.draw_image img 0 0
        );
      I.set_src img state.Reducer.choosed_image
    end
