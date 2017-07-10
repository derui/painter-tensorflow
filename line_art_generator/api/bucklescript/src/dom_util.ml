(*  *)
(* Abstract type for the DOM *)
type dom
(* Abstract type for an node *)
type node

type blob = Bs_fetch.blob

external dom : dom = "document" [@@bs.val]
(* binding document.createElement *)
external create_element : dom -> string -> node = "createElement" [@@bs.send]

(* binding document.createTextNode *)
external create_text_node : dom -> string -> node = "createTextNode" [@@bs.send]

external get_by_id : dom -> string -> node option =
  "getElementById" [@@bs.send] [@@bs.return null_to_opt]

(* binding FormData *)
module FormData = struct
  type t

  external create : unit -> t = "FormData" [@@bs.new]
  external append : string -> blob -> unit = "" [@@bs.send.pipe:t]
end

module Event = struct
  class type _t = object
    method preventDefault: unit -> unit
    method stopPropagation: unit -> unit
  end [@bs]
  type t = _t Js.t

  type 'a handler = t -> 'a
end

module Event_type = struct
  type t =
      DOMContentLoaded
    | Submit

  let to_string = function
    | DOMContentLoaded -> "DOMContentLoaded"
    | Submit -> "submit"
end

module Node = struct
  type t = node

  (* binding  node.appendChild *)
  external append_child : t -> t -> t = "appendChild" [@@bs.send]
  external set_class_name : t -> string -> unit = "className" [@@bs.set]
  external get_class_name : t -> string = "className" [@@bs.get]
  external set_attribute : t -> string -> 'a -> unit = "setAttribute" [@@bs.send]
  external get_files : t -> blob array = "files" [@@bs.get]

end

(* Binding for add_event_handler *)
external add_event_handler_ : dom -> string -> 'a Event.handler -> unit = "addEventHandler" [@@bs.send]

let add_event_handler el etype handler =
  let type_ = Event_type.to_string etype in
  add_event_handler_ el type_ handler
