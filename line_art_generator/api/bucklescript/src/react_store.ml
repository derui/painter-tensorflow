(*
  Define state store for flux infrastructure. This module is functor with State module that is defined the state on flux infrastructure.
*)

module type TYPE = sig
  type t
end

(* An interface of store module made from functor.
   Store is often that fully mutable 
 *)
module type S = sig
  type t
  type state

  val make: state -> t
  val save: t -> state -> t
  val get: t -> state
  val subscribe: t -> (unit -> unit) -> (t * (unit -> unit))
end

(* The functor to make the store *)
module Make(T:TYPE) : S with type state = T.t = struct
  type state = T.t
  module M = Map.Make(struct
                 type t = int
                 let compare = Pervasives.compare
               end)
  type t = {
      mutable state: T.t;
      mutable subscripters: (unit -> unit) M.t;
      mutable subscription_id: int;
    }

  let make state = {state = state; subscripters = M.empty; subscription_id = 0}
  let save t state =
    t.state <- state;
    M.iter (fun _ f -> f ()) t.subscripters;
    t
  let get t = t.state
  let subscribe t s =
    let next_id = succ t.subscription_id in
    t.subscripters <- M.add next_id s t.subscripters;
    t.subscription_id <- next_id;
    (t, fun () -> t.subscripters <- M.remove next_id t.subscripters)
end
