(* Define form component *)

module R = React

(* Property for file component *)
type prop = {
  state: Reducer.state;
  dispatcher: Dispatch.t;
}

external form_prop :
  ?className: string ->
  ?onSubmit: (('a, 'b) R.SyntheticEvent.t -> unit) ->
  unit -> _ = "" [@@bs.obj]

external img_prop :
  ?src: string ->
  unit -> _ = "" [@@bs.obj]

type state = unit

let make_image prop =
  let src = match prop.state.Reducer.generated_image with
    | None -> ""
    | Some s -> s in
  R.img (img_prop ~src ()) [||]

let render props _ _ =
  R.div (R.props ~className:"tp-ImagePreviewer_GeneratedImage" ()) [| make_image props |]

let t = R.createComponent render () (React.make_class_config ())
