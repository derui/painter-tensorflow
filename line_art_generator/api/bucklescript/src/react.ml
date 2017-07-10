(*

  Very thin React wrapper.

  Usage:
   - create element
     element "tag" any [||]
   - create component
     let component = create_component (fun p s _ -> text "foo") iniital_state in
     React.component component prop [||]

   - You can use pre-defined elements: div, span, a, input ...
*)
[%%bs.raw{|
import _React from 'createElement';
import _createReactClass from 'create-react-class';

function _createElement (clazz, props, children) {
  return _React.createElement(clazz, props, ...children);
}

function _createClass (fn, initialState) {
  return _createReactClass({
    getInitialState: function () {
      return { state: initialState };
    },

    render: function () {
      return fn(this.props, this.state.state, state => this.setState({ state }))
    }
  });
}
|}]

type element
type ('props, 'state) component

type 'state setStateFn = 'state -> unit
type ('props, 'state) renderFn = 'props -> 'state -> 'state setStateFn -> element
external createComponent_ : ('props, 'state) renderFn -> 'state -> ('props, 'state) component = "_createClass" [@@bs.val]

external createComponentElement_ : ('props, 'state) component -> 'props -> element array -> element = "_createElement" [@@bs.val]
external createBasicElement_ : string -> 'a Js.t -> element array -> element = "_createElement" [@@bs.val]

(* Needed so that we include strings and elements as children *)
external text : string -> element = "%identity"

(*
 * We have to do this indirection so that BS exports them and can re-import them
 * as known symbols. This is less than ideal.
 *)
let createComponent = createComponent_
let element = createBasicElement_

(* Event of React *)
module SyntheticEvent = struct
  class type _t = object
    method preventDefault: unit -> unit
    method stopPropagation: unit -> unit
    method bubbles: bool
    method cancelable: bool
    method currentTarget: Dom_util.Node.t
    method defaultPrevented: bool
    method eventPhase: int
    method isTrusted: bool
    method nativeEvent: Dom_util.Event.t
    method isDefaultPrevented: unit -> bool
    method isPropagationStopped: unit -> bool
    method target: Dom_util.Node.t
    method timeStamp: int
    method type_: string
  end [@bs]
  type t = _t Js.t
end

(* Define common prop object. *)
external props :
  ?onClick  : (SyntheticEvent.t -> unit) ->
  ?onChange : (SyntheticEvent.t -> unit) ->
  ?onSubmit : (SyntheticEvent.t -> unit) ->
  ?href     : string ->
  ?type_    : string ->
  ?value    : string ->
  unit -> _ =
  "" [@@bs.obj]

(* Ignore function currying with external function *)
let div props children = createBasicElement_ "div" props children
let span props children = createBasicElement_ "span" props children
let a props children = createBasicElement_ "a" props children
let button props children = createBasicElement_ "button" props children
let input props children = createBasicElement_ "input" props children
let form props children = createBasicElement_ "form" props children
let label props children = createBasicElement_ "label" props children
let p props children = createBasicElement_ "p" props children

let component comp = createComponentElement_ comp

(* -- *)

external render : element -> Dom_util.Node.t -> unit = "" [@@bs.module "react-dom"]
