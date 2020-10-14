# this is the tutorial

## 写在前面

* 如果是使用windows系统，则可以跟随tutorial顺畅地完成sphinx-quickstart之后还能和read the docs联动。

* 但如果你觉得官方文档还是太麻烦，且你有以下需求

  * 想使用中文和英文一起写文档
  * 想使用普通的markdown文档来生成html

  那么不妨采用本文档下面描述的解决方案

created in 2020/10/11

我必须承认这套sphinx和read the docs的联动在MAC上出现了奇怪的问题

所以我目前在MAC的解决方法是先用windows生成相关的文件夹，然后把这个文件夹（docs）复制到我想要加入文档的repo里面

基于这一解决方案，下面将给出详细的配置步骤

## 配置python环境

* 下载我放在项目组TAPD里的这个repo，并且创建一个用来生成文档的python虚拟环境（本经验使用的是python3.8.5）
* activate你想要用来编辑项目文档的python虚拟环境，并在本项目根目录下`pip install -r requirements.txt`

## 现在配置你的项目repo

* 把本repo的/docs文件夹完整地复制到你的repo的master分支的根目录下（注意必须是master分支，且必须是根目录，否则无法和read the docs联动，此后描述中使用‘/’表示项目根目录）（尤其是github更新后新创建的repo主分支默认名为main，特别注意！）

* （此后操作均在自己的repo里面进行）

* 进入docs文件夹，make.bat和makefile是已经用sphinx生成好的工具，不要触碰，/docs/build/是sphinx用于输出的文件夹，里面的内容不要手动修改

* 进入/docs/source/文件夹，应该有：

  * conf.py
  * Index.rst
  * 以及一系列自己写的*内容文件*（就是一堆markdown）

* 修改conf.py里的下述内容

  * project 现在是'learn_sphinx'，请把他改成你的项目名称
  * cpoyright 请改成你的年份和id
  * author改成你的id
  * release改成你的项目版本号（人为规定即可）
  * 其余保持不变

* 当你想要添加新的*内容文件*时，需要：

  * 把新添加的 .md 结尾的文件放到/docs/source/ 文件夹里

  * 使用vscode（反正只要不要用windows自带的记事本就行）打开文件/docs/index.rst，在第一长串等号下面是用rst语法描述的文档结构（就是那个toctree缩进里），目前文档结构里只有一个*内容文件*（就是本文件tutorial.md）如果以后想要向文档中加入更多*内容文件*，就仿照着在下面列出即可
  * 每次修改docs/index.rst之后，把它复制一份放到/docs/source/里面，替代掉老旧的/docs/source/index.rst

* 当你想要基于内容文件生成网页html的时候

  * 进入到/docs/文件夹（也就是makefile的同级位置）

  * 运行指令`make html`

* 成功完成后可以在docs/build/html/里面看到生成的html，为了更为优雅的体验，可以使用自己的浏览器打开其中的index.html进行阅读

## 和 read the docs 联动

* 去read the docs注册自己的read the docs账户（这里推荐使用自己的github账户来注册）
* 把自己的read the docs账户和github账户关联成功后，可以跟随read the docs 网页提示import自己的git repository到自己的read the docs repository
* 只要你的项目根目录下有按照前文搞定的docs文件夹，那么稍等片刻你就可以进入你的read the docs repository，在右上角会有一个绿色的按钮（View Docs），点开就是python官方风格的文档页面了，并且这个文档已经托管在readthedocs的服务器上，所有人都可以阅读。
* 此后每当你更新你的github repo时，readthedocs会自动同步，完全不需要你担心。