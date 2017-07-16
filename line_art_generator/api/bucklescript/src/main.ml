module D = Bs_dom_wrapper

class type _image_item =
  object
    method url : string
    method similarity : float
  end [@bs]
type image_item = _image_item Js.t

module Image_item = struct
  type t = image_item

  let from_json json =
    let module JJ = Js.Json in
    match JJ.classify json with
    | JJ.JSONObject obj -> begin
        let url = match Js.Dict.get obj "url" with
          | None -> raise (Js.Exn.raiseError "image item should have 'url'")
          | Some s -> begin
              match JJ.decodeString s with 
              | None -> raise (Js.Exn.raiseError "image item's url should be string")
              | Some s -> s
            end
        in
        let similarity = match Js.Dict.get obj "similarity" with
          | None -> raise (Js.Exn.raiseError "image item should have 'similarity'")
          | Some s -> begin
              match JJ.decodeNumber s with
              | None -> raise (Js.Exn.raiseError "image item's similarity should be number")
              | Some s -> s
            end
        in
        [%bs.obj { url = url; similarity = similarity }]
      end
    | _ -> raise (Js.Exn.raiseError "similarity response must be json object")
end

let () =
  (* let send_request file =  *)
  (*   let form_data = Dom_util.FormData.create () in *)
  (*   form_data |> Dom_util.FormData.append "file" file; *)

  (*   let open Bs_fetch in *)
  (*   fetchWithInit "/api/similarity-search" (RequestInit.make ~method_:Post ()) *)
  (*   |> Js.Promise.then_ (fun res -> *)
  (*     if Response.ok res then Response.json res else *)
  (*       res |> Response.statusText |> failwith |> Js.Promise.reject *)
  (*   ) *)
  (*   |> Js.Promise.then_ (fun json -> *)
  (*     let ret = match Js.Json.classify json with *)
  (*       | Js.Json.JSONObject obj -> *)
  (*          Js.Dict.get obj "similarities" |> similarities_to_images *)
  (*          |> Array.iter (fun _ -> ()) *)
  (*       | _ -> Js.log json *)
  (*     in *)
  (*     Js.Promise.resolve ret *)
  (*   ) *)
  (* in *)

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
