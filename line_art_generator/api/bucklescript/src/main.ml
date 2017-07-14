
class type _image_item = object
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
        | Some s -> begin match JJ.decodeString s with 
          | None -> raise (Js.Exn.raiseError "image item's url should be string")
          | Some s -> s
        end
      in
      let similarity = match Js.Dict.get obj "similarity" with
        | None -> raise (Js.Exn.raiseError "image item should have 'similarity'")
        | Some s -> begin match JJ.decodeNumber s with
          | None -> raise (Js.Exn.raiseError "image item's similarity should be number")
          | Some s -> s
        end
      in
      [%bs.obj { url = url; similarity = similarity }]
    end
    | _ -> raise (Js.Exn.raiseError "similarity response must be json object")
end

let create_card (item : image_item )=
  let module D = Dom_util in
  let div () = D.(create_element dom "div") in
  let card = div ()
  and card_media = div ()
  and card_text = div ()
  and img = D.(create_element dom "img")
  and text_node = D.(string_of_float item##similarity |> create_text_node dom) in

  D.Node.set_class_name img "image-container__image";
  D.Node.set_attribute img "src" item##url;

  D.Node.set_class_name card "mdl-card";
  D.Node.set_class_name card_media "mdl-card__media";
  D.Node.set_class_name card_text "mdl-card__supporting-text";

  D.Node.append_child card_text text_node |> ignore;
  D.Node.append_child card card_media |> ignore;
  D.Node.append_child card card_text |> ignore;

  card

let create_image_container item =
  let module D = Dom_util in
  let container = D.(create_element dom "div")
  and card = create_card item in

  D.Node.set_class_name container "image-container__item mdl-cell mdl-cell--3-col";
  D.Node.append_child container card |> ignore;

  container

let similarities_to_images = function
  | None -> [||]
  | Some obj -> match Js.Json.classify obj with
    | Js.Json.JSONArray ary ->
       Array.map Image_item.from_json ary |> Array.map create_image_container
    | _ -> [||]

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
    Dom_util.get_by_id Dom_util.dom "container" >>= fun c ->
    let store = Dispatch.Store.make {Reducer.file = ""} in
    let dispatcher = Dispatch.make ~store ~reducer:Reducer.reduce in
    let render c =
      React.render (React.component Component_main.t {
                        Component_main.state = Dispatch.Store.get store;
                        dispatcher = dispatcher
                      } [| |]) c in
    render c |> Option.return
  in 

  Dom_util.(add_event_handler dom Event_type.DOMContentLoaded ready_callback)
