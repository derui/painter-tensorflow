(* Define main container component *)

module R = React

type prop = {
  state: Reducer.state;
  dispatcher: Dispatch.t;
}

type state = unit

let previewer = (Component_previewer.t ())
let render prop _ _ =
  R.div (R.props ()) [|
      R.component Component_uploader.t {
          Component_uploader.state = prop.state;
          dispatcher = prop.dispatcher;
        } [||];

      R.component previewer {
          Component_previewer.state = prop.state;
          dispatcher = prop.dispatcher;
          height = 512;
          width = 512;
        } [||];
    |]

let t = R.createComponent render () (React.make_class_config ())
