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
var _React = require('react');
var _createReactClass = require('create-react-class');

function _createElement (clazz, props, children) {
  return _React.createElement(clazz, props, ...children);
}

function _createClass (fn, initialState, config) {
  return _createReactClass({
    getInitialState: function () {
      return { state: initialState };
    },

    componentWillReceiveProps: function(newProps) {
      if (config && config.willReceiveProps) {
        config.willReceiveProps(this.props, this.state.state, newProps, state => this.setState({state}));
      }
    },

    shouldComponentUpdate: function(props, state) {
      if (config && config.shouldUpdate) {
        return config.shouldUpdate(this.props, this.state.state, props, state.state);
      }
      return true;
    },

    componentDidUpdate: function() {
      if (config && config.didUpdate) {
        config.didUpdate(this.props, this.state.state, state => this.setState({state}));
      }
    },

    componentDidMount: function() {
      if (config && config.didMount) {
        return config.didMount(this.props, this.state.state, state => this.setState({state}));
      }
    },

    componentWillMount: function() {
      if (config && config.willMount) {
        return config.willMount(this.props, this.state.state, state => this.setState({state}));
      }
    },

    componentWillUnmount: function() {
      if (config && config.willUnmount) {
        return config.willUnmount(this.props, this.state.state);
      }
    },

    render: function () {
      return fn(this.props, this.state.state, state => this.setState({ state }))
    }
  });
}
|}]

module D = Bs_dom_wrapper

type element
type ('props, 'state) component
type 'state set_state_fn = 'state -> unit

type ('prop, 'state) should_update =
  'prop -> 'state -> 'prop -> 'state -> bool
type ('prop, 'state) mount = 'prop -> 'state -> 'state set_state_fn -> unit
type ('prop, 'state) unmount = 'prop -> 'state -> unit
type ('prop, 'state) receive_props = 'prop -> 'state -> 'prop -> 'state set_state_fn -> unit

(* make configuration object for component created from createComponent_ function *)
external make_class_config :
  ?shouldUpdate:('prop, 'state) should_update ->
  ?didUpdate:('prop, 'state) mount ->
  ?willReceiveProps:('prop, 'state) receive_props ->
  ?didMount:('prop, 'state) mount ->
  ?willMount:('prop, 'state) mount ->
  ?willUnmount:('prop, 'state) unmount ->
  unit -> _ = "" [@@bs.obj]

type ('props, 'state) render_fn = 'props -> 'state -> 'state set_state_fn -> element
external createComponent_ : ('props, 'state) render_fn -> 'state -> 'a Js.t -> ('props, 'state) component = "_createClass" [@@bs.val]

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
  class type ['a, 'b] _t =
    object
      method preventDefault: unit -> unit
      method stopPropagation: unit -> unit
      method bubbles: bool
      method cancelable: bool
      method currentTarget: 'a Dom.htmlElement_like
      method defaultPrevented: bool
      method eventPhase: int
      method isTrusted: bool
      method nativeEvent: 'b Dom.event_like
      method isDefaultPrevented: unit -> bool
      method isPropagationStopped: unit -> bool
      method target: 'a Dom.htmlElement_like
      method timeStamp: int
      method type_: string

      (* properties when event belongs Mouse Events *)
      method altKey: bool
      method button: int
      method buttons: int
      method clientX: int
      method clientY: int
      method ctrlKey: bool
      method getModifierState: int -> bool
      method metaKey: bool
      method pageX: int
      method pageY: int
      method relatedTarget: 'a Dom.htmlElement_like
      method screenX: int
      method screenY: int
      method shiftKey: bool

    end [@bs]
  type ('a, 'b) t = ('a, 'b) _t Js.t
end

(* Define common prop object. *)
external props :
  ?className: string ->
  ?onClick:(('a, 'b) SyntheticEvent.t -> unit) ->
  ?onChange:(('a, 'b) SyntheticEvent.t -> unit) ->
  ?onSubmit:(('a, 'b) SyntheticEvent.t -> unit) ->
  ?href:    string ->
  ?_type:   string ->
  ?value:   string ->
  ?defaultValue: string ->
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
let canvas props children = createBasicElement_ "canvas" props children
let img props children = createBasicElement_ "img" props children

let component comp = createComponentElement_ comp

(* -- *)

external render : element -> 'a Dom.node_like -> unit = "" [@@bs.module "react-dom"]
