(*
  Define state store for flux infrastructure. This module is functor with State module that is defined the state on flux infrastructure.
*)

module type TYPE = sig
  type t
end

(* An interface of store module made from functor *)
module type S = sig
  type t
  type state

  val make: state -> t
  val save: t -> state -> t
  val get: t -> state
  val subscribe: t -> (t -> state -> unit) -> t
end

(* The functor to make the store *)
module Make(T:TYPE) : S with type state = T.t = struct
  type state = T.t
  type t = {
    state: T.t;
    subscripters: (t -> T.t -> unit) array;
  }

  let make state = {state = state; subscripters = [||]}
  let save t state =
    let t' = {t with state} in
    Array.iter (fun f -> f t' state) t.subscripters;
    t'
  let get t = t.state
  let subscribe t s = {t with subscripters = Array.append [|s|] t.subscripters}
end
