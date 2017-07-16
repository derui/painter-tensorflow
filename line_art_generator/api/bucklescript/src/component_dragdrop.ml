(* Define drag&drop component *)

module D = Bs_dom_wrapper
module R = React

(* Property for file component *)
type prop = {
  state: Reducer.state;
  dispatcher: Dispatch.t;
}

external dandd_prop :
  ?className: string ->
  ?onDrop: ((D.Html.Input.t, D.Events.Drag_event.base) R.SyntheticEvent.t -> unit) ->
  unit -> _ = "" [@@bs.obj]

type state = unit

let on_drop props e =
  e##stopPropagation ();
  let files = D.Events.Drag_event.dataTransfer e##nativeEvent
              |> D.Data_transfer.files
  in
  match files |> Array.to_list with
  | [] -> ()
  | file :: _ -> 
     let dispatch = Dispatch.dispatch props.dispatcher in
     Actions.load_file dispatch file

let render props _ _ =
  R.div (dandd_prop ~className:"tp-ImageUploader_DragAndDrop"
           ~onDrop:(on_drop props) ()) [|
      R.text "Drop some image"
    |]

let t = R.createComponent render () (React.make_class_config ())
