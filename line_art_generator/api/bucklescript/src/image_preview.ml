(* Define function to update  *)
module D = Bs_dom_wrapper
module H = D.Html

let update_canvas state canvas =
  let context = canvas |> H.Canvas.getContext H.Types.Context_type.Context2D in
  if state.Reducer.choosed_image = "" then ()
  else begin
      let module I = H.Image in
      let module C = H.Canvas.Context in
      let img = I.create () in 
      I.setOnload img (fun _ ->
          context |> C.drawImage img 0 0
        );
      I.setSrc img state.Reducer.choosed_image
    end
