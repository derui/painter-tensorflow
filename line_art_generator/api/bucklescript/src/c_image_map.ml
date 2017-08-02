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
  ?ref: ('a -> unit) ->
  ?width: int ->
  ?height: int ->
  unit -> _ = "" [@@bs.obj]

type state = {
    image: string option;
    image_map: Reducer.image_map option;
  }
type inner_state = {
    mutable canvas: D.Dom.HtmlCanvasElement.t option;
  }

let update_canvas canvas props state =
  let open Option.Monad_infix in
  (canvas >>= (fun v ->
     state.image_map >>= fun im ->
     state.image >>= fun image ->

     let module I = D.Dom.HtmlImageElement in
     let module C = D.Dom.HtmlCanvasElement in
     let context = v |> C.getContext D.Dom.HtmlTypes.Context_type.Context2D in 
     let img = I.create () in 
     I.addEventListener "load" (fun _ ->
         let size = im.Reducer.scaled_size in

         context |> C.Context.clearRect 0 0 props.width props.height;
         context |> Reducer.(C.Context.drawImageWithDSize img 0 0 size.Size.width size.Size.height)
       ) img;
     I.setSrc img image |> Option.return
   )
  ) |> ignore

(* Current React FFI can not access "this" object of React component, so
 * we want to keep reference of canvas in component made.
 *)
let t () =
  (* closure to keep reference canvas element *)
  let inner_state = {canvas = None} in

  let render props _ _ =
    R.canvas (make_prop ~className: "tp-ImagePreviewer_ImageMapBase"
                ~width:props.width
                ~height:props.height
                ~ref:(fun v -> inner_state.canvas <- Some v)()) [||];
  in

  let should_update _ state _ new_state =
    match (state.image, new_state.image) with
    | (None, None) -> false
    | (None, _) -> true
    | (_, None) -> true
    | (Some v1, Some v2) -> v1 != v2
  in
  let did_update props state _ =
    update_canvas inner_state.canvas props state
  in
  let will_receive_props _ _ new_prop set_state =
    set_state {image = Some new_prop.state.Reducer.choosed_image;
               image_map = new_prop.state.Reducer.image_map;}
  in

  (* Avoid leaking canvas element. *)
  let will_unmount _ _ =
    inner_state.canvas <- None;
  in

  R.createComponent render {image = None;image_map = None}
    (React.make_class_config ~willReceiveProps:will_receive_props
       ~shouldUpdate:should_update
       ~didUpdate:did_update
       ~willUnmount:will_unmount ())
