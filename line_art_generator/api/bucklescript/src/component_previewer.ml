(* Define image previewer*)

module R = React
module D = Bs_dom_wrapper

(* Property for file component *)
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

type state = ()
type inner_state = {
    mutable canvas: D.Html.Canvas.t option
  }

let on_mousedown props _ =
  let module A = Actions in A.start_image_dragging () |> Dispatch.dispatch props.dispatcher 

let on_mouseup props _ =
  let module A = Actions in A.end_image_dragging () |> Dispatch.dispatch props.dispatcher 

let on_mousemove props e =
  let x = e##clientX
  and y = e##clientY in 
  let module A = Actions in  A.move_image x y |> Dispatch.dispatch props.dispatcher 

(* Current React FFI can not access "this" object of React component, so
 * we want to keep reference of canvas in component made.
 *)
let t () =
  (* closure to keep reference canvas element *)
  let inner_state = {canvas = None} in

  let render props _ _ =
    R.div (make_prop ~className:"tp-ImagePreviewer" ()) [|
        R.canvas (make_prop ~className: "tp-ImagePreviewer_Canvas"
                    ~onMouseDown:(on_mousedown props)
                    ~onMouseMove:(on_mousemove props)
                    ~onMouseUp:(on_mouseup props)
                    ~width:props.width
                    ~height:props.height
                    ~ref:(fun v -> inner_state.canvas <- Some v)()) [||]
      |]
  in

  let update_canvas canvas {state;_} =
    let f = Image_preview.update_canvas in
    let open Option.Monad_infix in
    (canvas >>= fun v -> f state v |> Option.return) |> ignore
  in

  let will_receive_props prop _ new_prop =
    if prop != new_prop then update_canvas inner_state.canvas new_prop else ()
  in

  let did_update prop _ = update_canvas inner_state.canvas prop in
  let did_mount prop _ = update_canvas inner_state.canvas prop in
  (* Avoid leaking canvas element. *)
  let will_unmount _ _ = inner_state.canvas <- None in

  R.createComponent render ()
    (React.make_class_config ~willReceiveProps:will_receive_props
       ~willUnmount:will_unmount
       ~didMount:did_mount
       ~didUpdate:did_update ())
