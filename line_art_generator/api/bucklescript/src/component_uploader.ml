(* Define form component *)

module R = React

(* Property for file component *)
type prop = {
  state: Reducer.state;
  dispatcher: Dispatch.t;
}

external form_prop :
  ?className: string ->
  ?onSubmit: (R.SyntheticEvent.t -> unit) ->
  unit -> _ = "" [@@bs.obj]

type state = unit

let on_submit _ e = e##preventDefault ()

let render props _ _ =
  R.form (form_prop ~className:"tp-ImageUploader" ~onSubmit:(on_submit props) ()) [|
      R.component Component_dragdrop.t {
          Component_dragdrop.state = props.state;
          dispatcher = props.dispatcher
        } [||];
      R.component Component_file.t {
          Component_file.state = props.state;
          dispatcher = props.dispatcher
        } [||];
    |]

let t = R.createComponent render () (React.make_class_config ())
