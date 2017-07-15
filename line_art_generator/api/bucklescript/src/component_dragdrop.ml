(* Define drag&drop component *)

module R = React

(* Property for file component *)
type prop = {
  state: Reducer.state;
  dispatcher: Dispatch.t;
}

external dandd_prop :
  ?className: string ->
  ?onDrop: (R.SyntheticEvent.t -> unit) ->
  unit -> _ = "" [@@bs.obj]

type state = unit

let on_drop props (e:R.SyntheticEvent.t) =
  e##stopPropagation ();
  let files = Dom_util.Node.get_files e##target in
  match files |> Array.to_list with
  | [] -> ()
  | file :: _ -> 
     let dispatch v f = Dispatch.dispatch props.dispatcher v f in
     Actions.load_file dispatch file

let render props _ _ =
  R.div (dandd_prop ~className:"tp-ImageUploader_DragAndDrop"
           ~onDrop:(on_drop props) ()) [|
      R.text "Drop some image"
    |]

let t = R.createComponent render () (React.make_class_config ())
