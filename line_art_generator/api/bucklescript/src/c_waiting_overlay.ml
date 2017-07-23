(* Define form component *)

module R = React

(* Property for file component *)
type prop = {
  state: Reducer.state;
  dispatcher: Dispatch.t;
}

type state = unit

let combine_class_name lst =
  List.filter (fun (_, p) -> p) lst
  |> List.map fst
  |> String.concat " "

let overlay_class_name props =
  let classes = [
      ("tp-WaitingOverlay", true);
      ("tp-WaitingOverlay-Active", props.state.Reducer.uploading);
    ] in
  combine_class_name classes

let spinner_class_name props =
  let classes = [
      ("tp-WaitingOverlay_Spinner", true);
      ("mdl-progress mdl-js-progress mdl-progress__indeterminate", true);
      ("is-active", props.state.Reducer.uploading);
    ] in
  combine_class_name classes

let render props _ _ =
  R.div (R.props ~className:(overlay_class_name props) ()) [|
      R.div (R.props ~className:"tp-WaitingOverlay_ProgressContainer" ()) [|
          R.div (R.props ~className:"tp-WaitingOverlay_Spacer" ()) [||];
          R.div (R.props ~className:"tp-WaitingOverlay_Progress" ()) [|
              R.div (R.props ~className:(spinner_class_name props) ()) [||];
            |];
          R.div (R.props ~className:"tp-WaitingOverlay_Message" ()) [|
              R.text "Now extracting..."
            |];
          R.div (R.props ~className:"tp-WaitingOverlay_Spacer" ()) [||];
        |];
    |]

let t = R.createComponent render () (React.make_class_config ())
