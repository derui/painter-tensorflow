(* Define image map *)

module R = React
module D = Bs_dom_wrapper

type prop = {
  state: Reducer.state;
  dispatcher: Dispatch.t;
  width: int;
  height: int;
}

external make_prop :
  ?className: string ->
  ?onMouseDown: (('a, D.Events.Mouse_event.base) R.SyntheticEvent.t -> unit) ->
  ?onMouseUp: (('a, D.Events.Mouse_event.base) R.SyntheticEvent.t -> unit) ->
  ?onMouseMove: (('a, D.Events.Mouse_event.base) R.SyntheticEvent.t -> unit) ->
  ?ref: ('a -> unit) ->
  ?width: int ->
  ?height: int ->
  unit -> _ = "" [@@bs.obj]

type state = {
    image_map: Reducer.image_map option;
  }

type inner_state = {
    mutable canvas: D.Html.Canvas.t option;
  }

let on_mousedown props _ =
  let module A = Actions in A.start_image_dragging () |> Dispatch.dispatch props.dispatcher 

let on_mouseup props _ =
  let module A = Actions in A.end_image_dragging () |> Dispatch.dispatch props.dispatcher 

let on_mousemove props e =
  if props.state.Reducer.dragging then
    let x = e##clientX
    and y = e##clientY in 
    let module A = Actions in  A.move_image x y |> Dispatch.dispatch props.dispatcher 
  else ()


let update_canvas canvas state =
  let open Option.Monad_infix in
  (canvas >>= fun c ->
   state.image_map >>= fun im ->
   let module H = D.Html in
   let module C = H.Canvas.Context in
   let ctx = c |> H.Canvas.getContext H.Types.Context_type.Context2D in
   C.setStrokeStyle ctx "rgb(0,255,0)";
   let s = im.Reducer.selector_size in
   let x, y = im.Reducer.selector_position in
   ctx |> C.strokeRect x y s.Reducer.Size.width s.Reducer.Size.height;
   Option.return ()
  ) |> ignore

(* Current React FFI can not access "this" object of React component, so
 * we want to keep reference of canvas in component made.
 *)
let t () =
  (* closure to keep reference canvas element *)
  let inner_state = {canvas = None} in

  let render props _ _ =
    R.canvas (make_prop ~className: "tp-ImagePreviewer_ImageMapSelector"
                ~onMouseDown:(on_mousedown props)
                ~onMouseMove:(on_mousemove props)
                ~onMouseUp:(on_mouseup props)
                ~width:props.width
                ~height:props.height
                ~ref:(fun v -> inner_state.canvas <- Some v)()) [||]
  in

  let should_update _ state _ new_state = state <> new_state in
  let will_receive_props _ _ new_prop set_state =
    set_state {image_map = new_prop.state.Reducer.image_map}
  in
  let did_update _ state _ =
    update_canvas inner_state.canvas state
  in

  (* Avoid leaking canvas element. *)
  let will_unmount _ _ =
    inner_state.canvas <- None;
  in

  R.createComponent render {image_map = None}
    (React.make_class_config
       ~willReceiveProps:will_receive_props
       ~didUpdate:did_update
       ~shouldUpdate:should_update
       ~willUnmount:will_unmount ())
