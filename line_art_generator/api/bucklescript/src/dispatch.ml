module Store = (React_store.Make(struct type t = Reducer.state end))

include React_dispatch.Make(Store)(Actions)

