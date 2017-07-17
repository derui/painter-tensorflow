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
  ?onMouseMove: ((Dom._baseClass, D.Events.Mouse_event.base) R.SyntheticEvent.t -> unit) ->
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
    let rect = D.Html.Element.getBoundingClientRect e##target in
    let x = e##clientX - (D.Dom_rect.left rect)
    and y = e##clientY - (D.Dom_rect.top rect) in 
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
   let rect = H.Canvas.getBoundingClientRect c in
   let s = im.Reducer.selector_size in
   let x, y = im.Reducer.selector_position in
   ctx |> C.clearRect 0 0 (D.Dom_rect.width rect) (D.Dom_rect.height rect);
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

  let should_update prop state new_prop new_state =
    if prop.state.Reducer.dragging <> new_prop.state.Reducer.dragging then
      true
    else
      match (state.image_map, new_state.image_map) with
      | None, None -> false
      | None, _ | _, None -> true
      | Some v1, Some v2 -> v1 <> v2
  in
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
