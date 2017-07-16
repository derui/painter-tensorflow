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
                    ~width:props.width
                    ~height:props.height
                    ~ref:(fun v -> inner_state.canvas <- Some v)()) [||];
      |]
  in

  R.createComponent render () (React.make_class_config ())
