(* Define image previewer*)

module R = React
module D = Bs_dom_wrapper
module H = D.Html

(* Property for file component *)
type prop = {
  state: Reducer.state;
  dispatcher: Dispatch.t;
  width: int;
  height: int;
}

external make_prop :
  ?className: string ->
  ?ref: ('a -> unit) ->
  ?width: int ->
  ?height: int ->
  unit -> _ = "" [@@bs.obj]

type state = {
    image: string option;
    image_map: Reducer.image_map option;
  }
type inner_state = {
    mutable canvas: D.Html.Canvas.t option
  }

let calc_original_scale v image_map =
  let factor = image_map.Reducer.scaling_factor in 
  float_of_int v |> fun v -> factor *. v |> floor |> int_of_float

let calc_original_size image_map =
  let size = image_map.Reducer.selector_size in
  {Reducer.Size.width = calc_original_scale size.Reducer.Size.width image_map;
   height = calc_original_scale size.Reducer.Size.height image_map}

let calc_original_pos image_map =
  let x, y = image_map.Reducer.selector_position in
  (calc_original_scale x image_map, calc_original_scale y image_map)

let update_canvas canvas state prop =
  let open Option.Monad_infix in
  (canvas >>=
     fun c -> state.image >>=
     fun image -> state.image_map >>=
     fun im -> 

     let module I = H.Image in
     let module C = H.Canvas.Context in
     let context = c |> H.Canvas.getContext H.Types.Context_type.Context2D in 
     let img = I.create () in 
     I.setOnload img (fun _ ->
         let size = calc_original_size im in
         let x,y = calc_original_pos im in

         context |> Reducer.(C.drawImageWithSSize img x y size.Size.width size.Size.height
                               0 0 prop.width prop.height);
         if not prop.state.Reducer.dragging then
           (* Update stripped image *)
           Lwt.async (fun () -> 
               let data = c |> H.Canvas.toDataURL "image/png" in
               Dispatch.dispatch prop.dispatcher (Actions.save_stripped_image data) |> Lwt.return
             )
         else ()
       );
     I.setSrc img image |> Option.return
  ) |> ignore

(* Current React FFI can not access "this" object of React component, so
 * we want to keep reference of canvas in component made.
 *)
let t () =
  (* closure to keep reference canvas element *)
  let inner_state = {canvas = None} in

  let render props _ _ =
    R.canvas (make_prop ~className: "tp-ImagePreviewer_Canvas"
                ~width:props.width
                ~height:props.height
                ~ref:(fun v -> inner_state.canvas <- Some v)()) [||];
  in

  let should_update prop state new_prop new_state =
    (prop.state.Reducer.dragging <> new_prop.state.Reducer.dragging)
    || not (Option.equal state.image new_state.image)
    || not (Option.equal state.image_map new_state.image_map)
  in

  let will_receive_props _ _ new_prop set_state =
    set_state {image = Some new_prop.state.Reducer.choosed_image;
               image_map = new_prop.state.Reducer.image_map;
      }
  in

  let did_update prop state _ =
    update_canvas inner_state.canvas state prop
  in

  let will_unmount _ _ = inner_state.canvas <- None in 

  R.createComponent render {image = None; image_map = None;}
    (React.make_class_config
       ~willReceiveProps:will_receive_props
       ~shouldUpdate:should_update
       ~didUpdate:did_update
       ~willUnmount:will_unmount
       ())
