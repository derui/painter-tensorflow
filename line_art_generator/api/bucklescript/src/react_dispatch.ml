(*
  React_dispatch module declares dispatcher for React component that like redux.

  Usage:

  type action = Action

  module D = React_dispatch.Make(struct
    type t = action
    let to_string = function
    | Action -> "action"
  end)
  (struct
    type state = string
    let reduce state = function
    | Action -> "s"
  end)

  let store = D.make ""
  let store = D.add_describe store (fun _ -> ()) in
  D.dispatch store v (fun v -> Action)

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
  type state
  type describe = state -> unit

  val make: state -> t
  val dispatch : t -> 'a -> ('a -> action) -> t
  val add_describe: t -> (state -> unit) -> t
end

module Make(A:Action)(R: Reducer with type action := A.t)
  : (S with type action := A.t and type state := R.state) =
struct
  type describe = R.state -> unit
  type t = {
    state: R.state;
    describes: describe array;
  }

  let make v = {state = v; describes = [||]}
  let dispatch t v f =
    let new_state = f v |> R.reduce t.state in
    let t = {t with state = new_state} in
    Array.iter (fun f -> f new_state) t.describes;
    t

  let add_describe t describe =
    {t with describes = Array.append [|describe|] t.describes}

end

