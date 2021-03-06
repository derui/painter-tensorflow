module B = Bs_webapi
module D = Bs_dom_wrapper

type t =
    StartFileLoading of string
  | EndFileLoading of (string * int * int)
  | StartImageDragging
  | MoveImage of (int * int)
  | EndImageDragging
  | SaveStrippedImage of string
  | StartImageUploading
  | EndImageUploading
  | UploadedImage of string

let to_string = function
  | StartFileLoading _ -> "start_file_loading"
  | EndFileLoading _ -> "end_file_loading"
  | StartImageDragging -> "end_image_dragging"
  | MoveImage _ -> "move_image"
  | EndImageDragging -> "end_image_dragging"
  | SaveStrippedImage _ -> "save_stripped_image"
  | StartImageUploading -> "start_image_uploading"
  | UploadedImage _ -> "uploaded_image"
  | EndImageUploading -> "end_image_uploading"

(* Actions for image dragging *)
let start_image_dragging () = StartImageDragging
let end_image_dragging () = EndImageDragging
let move_image x y = MoveImage (x, y)
(* Action for save partial image *)
let save_stripped_image image = SaveStrippedImage image

module Form_data = D.Form_data.Make(struct type t = Bs_fetch.formData end)

(* Action to upload stripped image *)
let upload_image dispatch image =
  let worker, wakeup = Lwt.task () in
  let module D = Bs_dom_wrapper in
  let form_data = Form_data.create () in
  form_data |> Form_data.appendString "image" image;

  let open Bs_fetch in
  fetchWithInit "/api/generate_image" (RequestInit.make
                                         ~method_:Post
                                         ~body:(BodyInit.makeWithFormData form_data)
                                         ())
  |> Js.Promise.then_ (fun res ->
         if Response.ok res then Response.blob res else
           res |> Response.statusText |> failwith |> Js.Promise.reject
       )
  |> Js.Promise.then_ (fun blob ->
         D.URL.t |> D.URL.createObjectURL blob |> Js.Promise.resolve
       )
  |> Js.Promise.then_ (fun url ->
         dispatch (UploadedImage url) |> Js.Promise.resolve
       )
  |> Js.Promise.then_ (fun _ ->
         Lwt.wakeup wakeup () |> Js.Promise.resolve
       )
  |> Js.Promise.catch (fun _ ->
       Lwt.(fail_with "error occured" |> ignore_result) |> Js.Promise.resolve)
  |> ignore;

  Lwt.finalize (fun () -> worker)
    (fun () -> dispatch EndImageUploading |> Lwt.return)
  |> Lwt.ignore_result;

  dispatch StartImageUploading

(* Load image from file object *)
let load_file dispatch file =
  let open Bs_dom_wrapper in
  let name = File.name file in
  let reader = File_reader.create () in
  File_reader.setOnload reader (fun _ ->
      let result = File_reader.result reader in
      let img = D.Dom.HtmlImageElement.create () in
      D.Dom.HtmlImageElement.addEventListener "load" (fun _ ->
          let width = D.Dom.HtmlImageElement.width img
          and height = D.Dom.HtmlImageElement.height img in
          dispatch (EndFileLoading (result, width, height))
        ) img;
      D.Dom.HtmlImageElement.setSrc img result;
    );
  reader |> File_reader.readAsDataURL file;
  dispatch (StartFileLoading name)
