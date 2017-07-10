(* Simple interface of monad *)
module type TYPE = sig
  type 'a t

  val return : 'a -> 'a t
  val bind : 'a t -> ('a -> 'b t) -> 'b t
  val map: 'a t -> ('a -> 'b) -> 'b t
end

module type Infix = sig
  type 'a t
    (* alias for bind *)
  val (>>=) : 'a t -> ('a -> 'b t) -> 'b t
    (* alias for map *)
  val (>>|) : 'a t -> ('a -> 'b) -> 'b t
end

module type S = sig
  type 'a t

  val return : 'a -> 'a t
  val bind : 'a t -> ('a -> 'b t) -> 'b t
  val map : 'a t -> ('a -> 'b) -> 'b t

  include Infix with type 'a t := 'a t
  module Monad_infix : Infix with type 'a t := 'a t
end

module type Monad = sig
  module type TYPE = TYPE
  module type Infix = Infix
  module type S = S

  module Make(T:TYPE) : S with type 'a t := 'a T.t
end
