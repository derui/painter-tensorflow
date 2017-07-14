module Store = (React_store.Make(struct type t = Reducer.state end))

include React_dispatch.Make(Store)(struct
  type t = Actions.t
  let to_string = function
    | `ChangeFile _ -> "change_action"
end)

