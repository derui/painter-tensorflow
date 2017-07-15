(* Define image previewer*)

module R = React

(* Property for file component *)
type prop = {
  state: Reducer.state;
  dispatcher: Dispatch.t;
  on_update_canvas: Reducer.state -> Dom_util.Canvas_element.t -> unit;
  width: int;
  height: int;
}

external make_prop :
  ?className: string ->
  ?ref: ('a -> unit) ->
  ?width: int ->
  ?height: int ->
  unit -> _ = "" [@@bs.obj]

type state = ()
type inner_state = {
    mutable canvas: Dom_util.Canvas_element.t option
  }

let on_submit _ e = e##preventDefault ()

(* Current React FFI can not access "this" object of React component, so
 * we want to keep reference of canvas in component made.
 *)
let t () =
  (* closure to keep reference canvas element *)
  let inner_state = {canvas = None} in

  let render props _ _ =
    R.div (make_prop ~className:"tp-ImagePreviewer" ()) [|
        R.canvas (make_prop ~className: "tp-ImagePreviewer_canvas"
                    ~width:props.width
                    ~height:props.height
                    ~ref:(fun v -> inner_state.canvas <- Some v)()) [||]
      |]
  in

  let update_canvas canvas {on_update_canvas = f;state;_} =
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
