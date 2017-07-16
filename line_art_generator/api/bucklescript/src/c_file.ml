(* Define file component *)

module D = Bs_dom_wrapper
module R = React

(* Property for file component *)
type prop = {
    state: Reducer.state;
    dispatcher: Dispatch.t;
  }

type state = unit

let on_change props e =
  let target: D.Html.Input.input Dom.htmlElement_like = e##target in
  let file = D.Html.Input.files target in
  let file = file.(0) in
  let dispatch = Dispatch.dispatch props.dispatcher in
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
