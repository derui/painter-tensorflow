(* Define file component *)

module D = Bs_dom_wrapper
module R = React

(* Property for file component *)
type prop = {
    state: Reducer.state;
    dispatcher: Dispatch.t;
  }

type state = unit

external make_prop :
  ?className: string ->
  ?disabled: Js.boolean ->
  ?_type: string ->
  unit -> _ = "" [@@bs.obj]

let on_change props e =
  let target: D.Html.Input.input Dom.htmlElement_like = e##target in
  let file = D.Html.Input.files target in
  let file = file.(0) in
  let dispatch = Dispatch.dispatch props.dispatcher in
  Actions.load_file dispatch file

let label_style =
  let styles = [
      "mdl-button";
      "mdl-js-button";
      "mdl-button--raised";
    ] in
  String.concat " " styles

let whole_style =
  let styles = [
      "mdl-grid";
      "mdl-grid--no-spacing";
      "tp-ImageUploader_Operations";
    ] in
  String.concat " " styles

let button_style =
  let classes = [
      "mdl-button";
      "mdl-js-button";
      "mdl-button--raised";
    ] in
  String.concat " " classes

let render props _ _ =
  R.div (R.props ~className:whole_style ()) [|
      R.div (R.props ~className:"mdl-cell mdl-cell--8-col" ()) [|
          R.text props.state.Reducer.file_name
        |];
      R.div (R.props ~className:"mdl-cell mdl-cell--2-col" ()) [|
          R.label (R.props ~className:label_style ()) [|
              R.input (R.props ~className:"tp-ImageUploader_File"
                         ~_type:"file"
                         ~defaultValue:(props.state.Reducer.file_name)
                         ~onChange:(on_change props) ())
                [||];
              R.text "Select";
            |]

        |];
      R.div (R.props ~className:"mdl-cell mdl-cell--2-col" ()) [|
          R.button (make_prop ~className:button_style
                      ~disabled:(Js.Boolean.to_js_boolean (props.state.Reducer.file_name = ""))
                      ~_type:"submit"
                      ())
            [|R.text "upload";|];
        |];
    |]

let t = R.createComponent render () (React.make_class_config ())
