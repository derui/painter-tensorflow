(* Define main container component *)

module R = React

type prop = {
  state: Reducer.state;
  dispatcher: Dispatch.t;
}

type state = unit

let render prop _ _ =
  R.div (R.props ()) [|
      R.component C_waiting_overlay.t {
          C_waiting_overlay.state = prop.state;
          dispatcher = prop.dispatcher;
        } [||];

      R.component C_uploader.t {
          C_uploader.state = prop.state;
          dispatcher = prop.dispatcher;
        } [||];

      R.component C_previewer.t {
          C_previewer.state = prop.state;
          dispatcher = prop.dispatcher;
          height = 512;
          width = 512;
        } [||];
    |]

let t = R.createComponent render () (React.make_class_config ())
