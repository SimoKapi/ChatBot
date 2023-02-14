import tkinter as tk

def main():
    interactions = []
    width, height = ["600", "400"]

    win = tk.Tk()
    win.title("Chatbot")
    win.geometry(f"{width}x{height}+50+50")
    win.resizable(False, False)
    win["background"] = "#7796f2"

    label = tk.Label(win, text='Use the chatbot:')
    label.place(relx=0.5, rely=0.05, anchor='s')

    entry = tk.Entry(win, width=30)
    entry.place(relx=0.5, rely=0.2, anchor="s")

    button = tk.Button(win, text="Ask")
    button.place(relx=0.5, rely=0.5, anchor="s")
    #text = tk.Text(win, height=25)
    #text.place(relx=0.5, rely=.9, anchor="s")
    #text.pack()

    win.mainloop()

if __name__ == "__main__":
    main()