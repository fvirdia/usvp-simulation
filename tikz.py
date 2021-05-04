class TikzPlot:

    def __init__(self, legend_pos="north east", grid="both", width="\\columnwidth"):
        self.legend_pos = legend_pos
        self.grid = grid
        self.width = width
        self.figsize = [10, 6]
        self.axes_labels = None
        self.xmin = self.xmax = self.ymin = self.ymax = None
        self.legend_cell_align = "left"
        self.elements = []
        self.legends = []
        self.scale = 0.7
        pass

    def line(self, coords, color="blue", linestyle="solid", legend_label=None, thickness=1, axes_labels=None, marker=None):
        # thickness ultra thin, very thin, thin, semithick, thick, very thick and ultra thick.

        if linestyle == "--":
            linestyle = "dashed"
        elif linestyle == " ":
            linestyle = "only marks"

        if isinstance(axes_labels, (tuple, list)):
            self.axes_labels = list(axes_labels)

        if isinstance(thickness, (int, float)):
            if thickness == 1:
                thickness = "thick" # semithick
            elif thickness == 1.5:
                thickness = "thick"
            elif thickness == 2:
                thickness = "very thick"
            elif thickness > 2:
                thickness = "ultra thick"
            elif thickness == 0.5:
                thickness = "thin"
            elif thickness < 0.5:
                thickness = "very thin"

        if marker == '|':
            marker = "square*"
        elif marker == 'd':
            marker = "diamond*"
        elif marker == '*':
            marker = "asterisk"
        elif marker == '+':
            marker = "+"
        elif marker == 'x':
            marker = "x, mark size=4pt"

        if color == "blue":
            color = "blue!90!black"
        elif color == "purple":
            color = "violet"
        elif color == "green":
            color = "green!60!black"
        elif color == "brown":
            color = "purple"

        src = "\\addplot[%s, %s, %s, x filter/.code={\\pgfmathparse{\\pgfmathresult+1.0}}%s] coordinates { " % (
            color, linestyle, thickness, "" if not marker else (", mark=%s" % marker)
        )

        for x, y in coords:
            src += "(%.4f, %.4f) " % (x, y)
        src += " };\n"

        if legend_label:
            self.legends.append(legend_label.replace('\\,', '\\ ').replace(',', '{,}').replace('[','{[}').replace(']','{]}'))
        else:
            self.legends.append('')

        self.elements.append(src)
        return src

    def save(self, filename, xmin=None, xmax=None, ymin=None, ymax=None, axes_labels=None, figsize=None, xticks=None, yticks=None, compile=True, **kwargs):
        if not self.xmin and xmin:
            self.xmin = xmin
        if not self.xmax and xmax:
            self.xmax = xmax
        if not self.ymin and ymin:
            self.ymin = ymin
        if not self.ymax and ymax:
            self.ymax = ymax
        if isinstance(axes_labels, (tuple, list)):
            self.axes_labels = list(axes_labels)
        if isinstance(figsize, (tuple, list)):
            self.figsize = list(figsize)

        src = "\\begin{tikzpicture}\n"
        src += "\\begin{axis}[\n"
        src += "\t/pgf/number format/.cd,\n"
        src += "\tfixed,\n"
        src += "\tgrid=%s,\n" % self.grid
        src += "\tscale=%s,\n" % self.scale
        # src += "\tlegend style={font=\\fontsize{12}{0}\\selectfont},\n"
        src += "\tlegend pos=%s,\n" % self.legend_pos
        if isinstance(self.axes_labels, list):
            if len(self.axes_labels) >= 1:
                src += "\txlabel=%s,\n" % self.axes_labels[0]
                # src += "\tx label style={font=\\fontsize{14}{0}\\selectfont},\n"
            if len(self.axes_labels) >= 2:
                src += "\tylabel=%s,\n" % self.axes_labels[1]
                # src += "\ty label style={font=\\fontsize{14}{0}\\selectfont},\n"
        src += "\twidth=%s,\n" % self.width
        src += "\theight=%.2f%s,\n" % (self.figsize[1]/self.figsize[0], self.width)
        if self.xmin:
            src += "\txmin = %.2f,\n" % self.xmin
        if self.xmax:
            src += "\txmax = %.2f,\n" % self.xmax
        if self.ymin:
            src += "\tymin = %.2f,\n" % self.ymin
        if self.ymax:
            src += "\tymax = %.2f,\n" % self.ymax
        if isinstance(xticks, int) and self.xmin and self.xmax:
            skip = (self.xmax - self.xmin)/(xticks+1)
            src += "\txtick = {" + ",".join([str(self.xmin + skip * (1+_)) for _ in range(xticks)]) + "},\n"
        if isinstance(yticks, int) and self.ymin and self.ymax:
            skip = (self.ymax - self.ymin)/(yticks+1)
            src += "\tytick = {" + ",".join([str(self.ymin + skip * (1+_)) for _ in range(yticks)]) + "},\n"

        src += "\tlegend cell align=%s,\n" % self.legend_cell_align
        src += "]\n"

        src += "".join(self.elements)
        src += "\\legend{" + ",".join(self.legends) + "}\n"

        src += "\\end{axis}\n\\end{tikzpicture}\n"

        with open(filename, "w") as f:
            f.write(src)


        path = '/'.join(filename.split('/')[:-1])
        fn = '.'.join(filename.split('/')[-1].split('.')[:-1])
        with open(f"{path}/{fn}.tex", "w") as f:
            tex = "\\documentclass[class=minimal,border=0pt]{standalone}\n"
            tex += "\\usepackage{amsmath,amsfonts,amssymb,stmaryrd,amsthm}\n"
            tex += "\\usepackage{tikz}\n"
            tex += "\\usepackage{pgfplots}\n"
            tex += "\\begin{document}\n"
            tex += "\\input{\\detokenize{%s}}\n" % filename.split('/')[-1]
            tex += "\\end{document}\n"
            f.write(tex)

        if compile:
            import os
            os.chdir(path)
            os.system(f'pdflatex {fn}.tex')
            # os.system(f'rm {fn}_.tex {fn}_.aux {fn}_.log {fn}_.synctex.gz texput.log')
            os.chdir("/".join([".." for _ in range(len(path.split("/")))]))


    def __iadd__(self, other):
        if not isinstance(other, TikzPlot):
            raise ValueError("Can only combine TikzPlot objects")
        self.elements = self.elements + other.elements
        self.legends = self.legends + other.legends
        return self
