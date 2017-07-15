(*  *)
(* Abstract type for the DOM *)
type dom

type blob = Bs_fetch.blob

(* Interface for File *)
module File = struct
  type t = blob

  external name: t -> string = "" [@@bs.get]
end

(* binding FormData *)
module FormData = struct
  type t

  external create : unit -> t = "FormData" [@@bs.new]
  external append : string -> blob -> unit = "" [@@bs.send.pipe:t]
end

(* Event interface. *)
module Event = struct
  type 'a t

  type ('a, 'b) handler = 'a t -> 'b
  external preventDefault: unit -> unit = "" [@@bs.send.pipe:'a t]
  external stopPropagation: unit -> unit = "" [@@bs.send.pipe:'a t]
end

module Event_type = struct
  type t =
    DOMContentLoaded
  | Submit

  let to_string = function
    | DOMContentLoaded -> "DOMContentLoaded"
    | Submit -> "submit"
end

(* Reference to EventTarget interface *)
module Event_target = struct
  type 'a t

  external add_event_listener: Event_type.t -> ('a, 'b) Event.handler -> unit = "addEventListener" [@@bs.send.pipe:'a t]
  external remove_event_listener: Event_type.t -> ('a, 'b) Event.handler -> unit = "removeEventListener" [@@bs.send.pipe:'a t]

end

(* Binding for ProgressEvent interface *)
module Progress_event = struct
  type 'a progress
  type 'a t = 'a progress Event.t

  external get_target: 'a t -> 'a Event_target.t = "target" [@@bs.get]
end

module Node = struct
  type node
  type t = node Event_target.t

  (* binding  node.appendChild *)
  external append_child : t -> t -> t = "appendChild" [@@bs.send]
  external set_class_name : t -> string -> unit = "className" [@@bs.set]
  external get_class_name : t -> string = "className" [@@bs.get]
  external set_attribute : t -> string -> 'a -> unit = "setAttribute" [@@bs.send]
  external get_files : t -> File.t array = "files" [@@bs.get]

  external get_value : t -> string = "value" [@@bs.get]
  external set_value : t -> string -> unit = "value" [@@bs.set]
end

module Image_element = struct
  type image
  type t = image Event_target.t

  external create: unit -> t = "Image" [@@bs.new]
  external set_src: t -> string -> unit = "src" [@@bs.set]
  external get_src: t -> string = "src" [@@bs.get]
  external set_onload: t -> (image Progress_event.t -> unit) -> unit = "onload" [@@bs.set]
end

module Canvas_context = struct
  type t

  (* binding drawImage *)
  external draw_image: Image_element.t -> int -> int -> unit = "drawImage" [@@bs.send.pipe:t]
end

module Canvas_element = struct
  type t

  (* binding getContext() method *)
  external get_context: string -> Canvas_context.t = "getContext" [@@bs.send.pipe:t]
end

(* Interface for FileReader *)
module FileReader = struct
  type file_reader
  type t = file_reader Event_target.t

  external get_result: t -> string = "result" [@@bs.get]
  external create: unit -> t = "FileReader" [@@bs.new]
  external set_onload: t -> (file_reader Progress_event.t -> unit) -> unit = "onload" [@@bs.set]
  external read_as_data_url: t -> File.t -> unit = "readAsDataURL" [@@bs.send]
end


(* Binding for add_event_handler on document *)
external _add_event_listener : dom -> string -> ('a, 'b) Event.handler -> unit = "addEventListener" [@@bs.send]

let add_event_listener el etype handler =
  let type_ = Event_type.to_string etype in
  _add_event_listener el type_ handler

external dom : dom = "document" [@@bs.val]
(* binding document.createElement *)
external create_element : dom -> string -> Node.t = "createElement" [@@bs.send]

(* binding document.createTextNode *)
external create_text_node : dom -> string -> Node.t = "createTextNode" [@@bs.send]

external get_by_id : dom -> string -> Node.t option =
  "getElementById" [@@bs.send] [@@bs.return null_to_opt]

external query_selector : dom -> string -> Node.t option =
  "querySelector" [@@bs.send] [@@bs.return null_to_opt]
