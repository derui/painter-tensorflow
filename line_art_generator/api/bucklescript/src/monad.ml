include Monad_intf

(* Functor to make the monad for TYPE *)
module Make(T: TYPE) : S with type 'a t := 'a T.t = struct

  let return = T.return
  let bind = T.bind
  let map = T.map

  module Monad_infix = struct
    let (>>=) = T.bind
    let (>>|) = T.map
  end
  include Monad_infix
end
