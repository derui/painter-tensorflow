(* Define file component *)

module R = React

(* Property for file component *)
type prop = {
    state: Reducer.state;
    dispatcher: Dispatch.t;
  }

type state = unit

let on_change props e =
  let file = Dom_util.Node.get_files e##target in
  let file = file.(0) in
  let dispatch v f = Dispatch.dispatch props.dispatcher v f in
  Actions.load_file dispatch file

let label_style =
  let styles = [
      "tp-ImageUploader_UploadButton";
      "mdl-button";
      "mdl-js-button";
      "mdl-button--raised";
    ] in 
  String.concat " " styles

let render props _ _ =
  R.label (R.props ~className:label_style ()) [|
      R.input (R.props ~className:"tp-ImageUploader_file"
                 ~_type:"file"
                 ~defaultValue:(props.state.Reducer.file_name)
                 ~onChange:(on_change props) ())
        [||];
      R.text "Upload"
    |] 

let t = R.createComponent render () (React.make_class_config ())
