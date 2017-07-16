(*
  React_dispatch module declares dispatcher for React component that like redux.

  Usage:

  type action = Action
  module A = struct
    type t = action
    let to_string = function
    | Action -> "action"
  end

  module R = struct
    type state = string
    let reduce state = function
    | Action -> "s"
  end

  module D = React_dispatch.Make(React_store.Make(struct type t = string end))(A)(R)

  let store = D.make ""
  let store = D.subscribe store (fun _ -> ()) in
  let dispatch = D.make store in 
  dispatch v (fun v -> Action)

*)

module type Action = sig
  type t

  (* Convert action to string *)
  val to_string : t -> string
end

module type Reducer = sig
  type action
  type state

  val reduce: state -> action -> state
end

module type S = sig
  type t
  type action
  type store
  type state
  type reducer = state -> action -> state

  val make : store:store -> reducer:reducer -> t
  val dispatch: t -> action -> unit
  val subscribe: t -> (unit -> unit) -> (unit -> unit)
end

module Make(Store: React_store.S)
  (A: Action)
  : (S with type action := A.t and type store := Store.t and type state := Store.state) =
struct
  type reducer = Store.state -> A.t -> Store.state
  type t = {
    store: Store.t;
    reducer: reducer;
  }

  (* This function return closure contained mutable store. *)
  let make ~store ~reducer = {store; reducer}

  let dispatch t v =
    let new_state = t.reducer (Store.get t.store) v in
    Store.save t.store new_state |> ignore

  let subscribe t subscription =
    let _, unsub = Store.subscribe t.store subscription in
    unsub
end

