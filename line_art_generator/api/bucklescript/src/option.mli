include Monad.S with type 'a t := 'a option
(* Monadic interface for option *)

(* utility function *)
val is_some: 'a option -> bool

(* Check equal option values *)
val equal: 'a option -> 'a option -> bool
