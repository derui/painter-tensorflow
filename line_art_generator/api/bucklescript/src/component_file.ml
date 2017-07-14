(* Define file component *)

module R = React

(* Property for file component *)
type prop = {
  state: Reducer.state;
  dispatcher: Dispatch.t;
}

type state = unit

let on_change props e =
  let value = Dom_util.Node.get_value e##target in
  Dispatch.dispatch props.dispatcher value (fun v -> `ChangeFile v)

let render props _ _ =
  R.input (R.props ~type_:"file" ~value:(props.state.Reducer.file) ~onChange:(on_change props) ()) [||]

let t = R.createComponent render () (React.make_class_config ())
