include Monad.S with type 'a t := 'a option
(* Monadic interface for option *)

(* utility function *)
val is_some: 'a option -> bool

