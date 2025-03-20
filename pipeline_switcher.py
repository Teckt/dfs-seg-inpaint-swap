import tkinter as tk
from redresser_utils import SocketClient, SocketServer


class FluxSwitchUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.s = SocketClient(port=7777)
        self.title("Flux Controller")
        self.geometry("200x100")

        btn_flux = tk.Button(self, text="Flux", command=lambda: self.send_command("t2i"))
        btn_flux.pack(pady=10)

        btn_flux_fill = tk.Button(self, text="Flux-fill", command=lambda: self.send_command("fill"))
        btn_flux_fill.pack(pady=10)

    def send_command(self, command):
        self.s.put(command)


def switch_to_fill_pipe():
    dispatcher = SocketClient(7777)
    dispatcher.put("fill", retry_wait=5, max_retries=99999999)


def switch_to_t2i_pipe():
    dispatcher = SocketClient(7777)
    dispatcher.put("t2i", retry_wait=5, max_retries=99999999)


if __name__ == "__main__":
    # app = FluxSwitchUI()
    # app.mainloop()

    receiver = SocketServer(7778)
      # sends commands to this socket