module D = Bs_dom_wrapper

let () =
  let module Store = React_store.Make(struct
                         type t = Reducer.state
                       end) in
  let ready_callback = fun _ ->
    let open Option.Monad_infix in
    let _ =
      D.document |> D.Nodes.Document.querySelector ".tp-Content" >>= (fun c ->
        let store = Dispatch.Store.make Reducer.empty in
        let dispatcher = Dispatch.make ~store ~reducer:Reducer.reduce in
        let render () =
          React.render (React.component C_main.t {
                            C_main.state = Dispatch.Store.get store;
                            dispatcher = dispatcher
                          } [| |]) c in

        let _ = Dispatch.subscribe dispatcher render in 
        render () |> Option.return
      )
    in ()
  in

  D.document |> D.Nodes.Document.addEventListener "DOMContentLoaded" ready_callback
