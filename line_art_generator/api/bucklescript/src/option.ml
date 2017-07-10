
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
