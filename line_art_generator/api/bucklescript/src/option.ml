
include Monad.Make(struct
  type 'a t = 'a option
  let return v = Some v
  let bind v f = match v with
    | None -> None
    | Some v' -> f v'

  let map v f = match v with
    | None -> None
    | Some v' -> return (f v')
end)

let is_some = function
  | None -> false
  | Some _ -> true

let equal v1 v2 =
  match (v1, v2) with
  | None, None -> true
  | None, _ | _, None -> false
  | Some v1, Some v2 -> v1 = v2
