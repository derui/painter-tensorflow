(* Define image previewer*)

module R = React
module D = Bs_dom_wrapper

(* Property for file component *)
type prop = {
    state: Reducer.state;
    dispatcher: Dispatch.t;
    width: int;
    height: int;
  }

type state = ()
let image_map = C_image_map.t ()
let image_area_selector = C_image_area_selector.t ()
let stripped_image = C_stripped_image.t ()

let render props _ _ =
  R.div (R.props ~className:"tp-ImagePreviewer" ()) [|
      R.div (R.props ~className:"tp-ImagePreviewer_BaseImageContainer" ())
        [| R.div (R.props ~className:"tp-ImagePreviewer_StrippedImage" ()) [|
               R.component stripped_image {
                   C_stripped_image.state = props.state;
                   dispatcher = props.dispatcher;
                   width = 512;
                   height = 512;
                 } [||]
             |];
           R.div (R.props ~className:"tp-ImagePreviewer_ImageMapContainer" ()) [|
               R.component image_map {
                   C_image_map.state = props.state;
                   dispatcher = props.dispatcher;
                   width = 256;
                   height = 256;
                 } [||];
               R.component image_area_selector {
                   C_image_area_selector.state = props.state;
                   dispatcher = props.dispatcher;
                   width = 256;
                   height = 256;
                 } [||];
             |];
        |];
      R.component C_generated_image.t {
          C_generated_image.state = props.state;
          dispatcher = props.dispatcher;
        } [||];
    |]

let t = R.createComponent render () (R.make_class_config ())
