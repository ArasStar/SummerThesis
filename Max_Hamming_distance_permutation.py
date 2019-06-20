{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Max_Hamming_distance_permutation.ipynb",
      "version": "0.3.2",
      "provenance": [],
      "collapsed_sections": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/ArasStar/SummerThesis/blob/master/Max_Hamming_distance_permutation.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "73Xs4KcIr1p-",
        "colab_type": "code",
        "outputId": "64159ed5-7f00-4ecf-f6aa-bb0154a0cbe5",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 548
        }
      },
      "source": [
        "%reload_ext autoreload\n",
        "%autoreload 2\n",
        "%matplotlib inline\n",
        "\n",
        "#DOWNLOAD THE DATA\n",
        "from google.colab import drive\n",
        "import zipfile\n",
        "\n",
        "#MOUNT GDRIVE: DOWNLOAD DATA AND FILTERED LABELS\n",
        "drive.mount('/content/gdrive',force_remount=True)\n",
        "print(\"mounted google drive\")\n",
        "\n",
        "## UNZIP ZIP\n",
        "#print (\"Uncompressing zip file\")\n",
        "#zip_ref = zipfile.ZipFile('/content/gdrive/My Drive/CheXpert-v1.0-small.zip', 'r')\n",
        "#zip_ref.extractall()\n",
        "#zip_ref.close()\n",
        "#print(\"downloaded files\")"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "error",
          "ename": "KeyboardInterrupt",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/ipykernel/kernelbase.py\u001b[0m in \u001b[0;36m_input_request\u001b[0;34m(self, prompt, ident, parent, password)\u001b[0m\n\u001b[1;32m    729\u001b[0m             \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 730\u001b[0;31m                 \u001b[0mident\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mreply\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msession\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrecv\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mstdin_socket\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m0\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    731\u001b[0m             \u001b[0;32mexcept\u001b[0m \u001b[0mException\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/jupyter_client/session.py\u001b[0m in \u001b[0;36mrecv\u001b[0;34m(self, socket, mode, content, copy)\u001b[0m\n\u001b[1;32m    802\u001b[0m         \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 803\u001b[0;31m             \u001b[0mmsg_list\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0msocket\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrecv_multipart\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmode\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcopy\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mcopy\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    804\u001b[0m         \u001b[0;32mexcept\u001b[0m \u001b[0mzmq\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mZMQError\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/zmq/sugar/socket.py\u001b[0m in \u001b[0;36mrecv_multipart\u001b[0;34m(self, flags, copy, track)\u001b[0m\n\u001b[1;32m    465\u001b[0m         \"\"\"\n\u001b[0;32m--> 466\u001b[0;31m         \u001b[0mparts\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrecv\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mflags\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcopy\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mcopy\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtrack\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mtrack\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    467\u001b[0m         \u001b[0;31m# have first part already, only loop while more to receive\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32mzmq/backend/cython/socket.pyx\u001b[0m in \u001b[0;36mzmq.backend.cython.socket.Socket.recv\u001b[0;34m()\u001b[0m\n",
            "\u001b[0;32mzmq/backend/cython/socket.pyx\u001b[0m in \u001b[0;36mzmq.backend.cython.socket.Socket.recv\u001b[0;34m()\u001b[0m\n",
            "\u001b[0;32mzmq/backend/cython/socket.pyx\u001b[0m in \u001b[0;36mzmq.backend.cython.socket._recv_copy\u001b[0;34m()\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/zmq/backend/cython/checkrc.pxd\u001b[0m in \u001b[0;36mzmq.backend.cython.checkrc._check_rc\u001b[0;34m()\u001b[0m\n",
            "\u001b[0;31mKeyboardInterrupt\u001b[0m: ",
            "\nDuring handling of the above exception, another exception occurred:\n",
            "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-1-0dd9ed9b54bf>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      8\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      9\u001b[0m \u001b[0;31m#MOUNT GDRIVE: DOWNLOAD DATA AND FILTERED LABELS\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 10\u001b[0;31m \u001b[0mdrive\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmount\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'/content/gdrive'\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mforce_remount\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     11\u001b[0m \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"mounted google drive\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     12\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/google/colab/drive.py\u001b[0m in \u001b[0;36mmount\u001b[0;34m(mountpoint, force_remount, timeout_ms)\u001b[0m\n\u001b[1;32m    182\u001b[0m       \u001b[0;31m# Not already authorized, so do the authorization dance.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    183\u001b[0m       \u001b[0mauth_prompt\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0md\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmatch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mgroup\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m+\u001b[0m \u001b[0;34m'\\n\\nEnter your authorization code:\\n'\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 184\u001b[0;31m       \u001b[0md\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msend\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0m_getpass\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mgetpass\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mauth_prompt\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m+\u001b[0m \u001b[0;34m'\\n'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    185\u001b[0m   \u001b[0md\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msendcontrol\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'z'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    186\u001b[0m   \u001b[0md\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mexpect\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34mu'Stopped'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/ipykernel/kernelbase.py\u001b[0m in \u001b[0;36mgetpass\u001b[0;34m(self, prompt, stream)\u001b[0m\n\u001b[1;32m    686\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_parent_ident\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    687\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_parent_header\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 688\u001b[0;31m             \u001b[0mpassword\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    689\u001b[0m         )\n\u001b[1;32m    690\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/ipykernel/kernelbase.py\u001b[0m in \u001b[0;36m_input_request\u001b[0;34m(self, prompt, ident, parent, password)\u001b[0m\n\u001b[1;32m    733\u001b[0m             \u001b[0;32mexcept\u001b[0m \u001b[0mKeyboardInterrupt\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    734\u001b[0m                 \u001b[0;31m# re-raise KeyboardInterrupt, to truncate traceback\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 735\u001b[0;31m                 \u001b[0;32mraise\u001b[0m \u001b[0mKeyboardInterrupt\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    736\u001b[0m             \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    737\u001b[0m                 \u001b[0;32mbreak\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7isIyCu_5gVa",
        "colab_type": "code",
        "outputId": "ffbf005f-5c3d-4b51-b734-3c0ce0873b93",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 51
        }
      },
      "source": [
        "import torch\n",
        "from sympy.utilities.iterables import multiset_permutations\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "from google.colab import files\n",
        "\n",
        "file_name_p_set = \"checkpoint_p_set.pt\"\n",
        "file_name_P_= \"checkpoint_P_.pt\"\n",
        "file_name_i = \"checkpoint_i.txt\"\n",
        "\n",
        "PATH_p_set = F\"gdrive/My Drive/summerthesis/permutation_set/{file_name_p_set}\" \n",
        "PATH_P_ = F\"gdrive/My Drive/summerthesis/permutation_set/{file_name_P_}\" \n",
        "PATH_i = F\"gdrive/My Drive/summerthesis/permutation_set/{file_name_i}\" \n",
        "\n",
        "\n",
        "use_cuda = True\n",
        "if use_cuda and torch.cuda.is_available():\n",
        "    device = torch.device('cuda')\n",
        "else:\n",
        "    print(\"CUDA didn't work\")\n",
        "    device = torch.device('cpu')\n",
        "\n",
        "    \n",
        "def pick_j(p_set, P_):\n",
        "  \n",
        "  D = torch.Tensor().to(device=device)\n",
        "  for p in p_set:\n",
        "    #hamming distances of ith for all other P'\n",
        "    D_i = torch.stack([(p != p_prime) for p_prime in P_[1:,:].t()],dim=1).sum(dim=0).view(1,-1).type(torch.FloatTensor).to(device=device)\n",
        "    D=torch.cat((D,D_i)).to(device = device)\n",
        "  \n",
        "  D = D.sum(dim=0).view(1,-1).to(device=device)\n",
        "  j = P_[0,torch.argmax(D).item()].item()#first row of P_ is index checking which indexed perm is max\n",
        "\n",
        "  return int(j)\n",
        "\n",
        "\n",
        "set_p = np.array([0,1,2,3,4,5,6,7,8])\n",
        "P_ = torch.stack([ torch.FloatTensor(p) for p in multiset_permutations(set_p)],dim=1).to(device=device)\n",
        "_,perm_size = P_.shape\n",
        "\n",
        "idx = torch.arange(perm_size).type(torch.FloatTensor).view(1,-1).to(device=device)\n",
        "P_=torch.cat((idx,P_))\n",
        "\n",
        "p_set = torch.Tensor().to(device=device)\n",
        "j=np.random.choice(perm_size)\n",
        "\n",
        "i = 0\n",
        "checkpoint = 1\n",
        "cardinality = 110\n",
        "\n",
        "while i <= cardinality :\n",
        "  \n",
        "  if checkpoint == 0:\n",
        "    \n",
        "    indice = (P_[0,:] == j).nonzero()\n",
        "    p_j = P_[1:,indice].view(1,-1).to(device=device)\n",
        "    p_set = torch.cat((p_set,p_j)).to(device=device)\n",
        "    P_ = torch.cat((P_[:,:indice] , P_[:,indice+1:]),dim=1).to(device=device)\n",
        "    \n",
        "  else:\n",
        "    p_set = torch.load(PATH_p_set)\n",
        "    P_ = torch.load(PATH_P_)#index is the first row\n",
        "    with open(PATH_i, \"r\") as text_file:\n",
        "      i =int(text_file.read())\n",
        "      \n",
        "    checkpoint=0\n",
        "    print(\"starting from checkpoint i=\",i)\n",
        "    \n",
        "  if i % 10 == 0:\n",
        "    print(\"i:\",i)\n",
        " \n",
        "  if i == cardinality :\n",
        "  \n",
        "    torch.save(p_set, PATH_p_set)\n",
        "    torch.save(P_, PATH_P_)\n",
        "\n",
        "    with open(PATH_i, \"w\") as text_file:\n",
        "      text_file.write(str(p_set.shape[0]))\n",
        "      \n",
        "    print('saved perm set to google drive')\n",
        "  \n",
        "  j = pick_j(p_set, P_)\n",
        "  i = i+1\n",
        "\n",
        "\n",
        "print(\"now checking uniqueness\")\n",
        "output = torch.unique(p_set,dim=0)\n",
        "print(\"p_set\",p_set.shape)\n",
        "print(\"output(uniqie row)\",output.shape)\n",
        "\n",
        "\n",
        "print(p_set)\n",
        "#P_ = torch.stack([torch.FloatTensor(p) for p in multiset_permutations(p_set)])"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "starting from checkpoint i= 100\n",
            "i: 100\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "DMNT360xoyjP",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        ""
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Ihos17OgvURN",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        ""
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "idPT9wJgysZk",
        "colab_type": "code",
        "outputId": "57de4164-e029-4ec5-966e-1e92edf31b25",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 68
        }
      },
      "source": [
        "##HOOOOOOOOP\n",
        "import torch\n",
        "import numpy as np\n",
        "from sympy.utilities.iterables import multiset_permutations\n",
        "\n",
        "\n",
        "use_cuda = True\n",
        "if use_cuda and torch.cuda.is_available():\n",
        "    device = torch.device('cuda')\n",
        "else:\n",
        "    print(\"CUDA didn't work\")\n",
        "    device = torch.device('cpu')\n",
        "    \n",
        "file_name_p_set = \"checkpoint_p_set.pt\"\n",
        "file_name_P_= \"checkpoint_P_.pt\"\n",
        "file_name_i = \"checkpoint_i.txt\"\n",
        "\n",
        "PATH_p_set = F\"/content/gdrive/My Drive/summerthesis/permutation_set/{file_name_p_set}\" \n",
        "PATH_P_ = F\"/content/gdrive/My Drive/summerthesis/permutation_set/{file_name_P_}\" \n",
        "PATH_i = F\"/content/gdrive/My Drive/summerthesis/permutation_set/{file_name_i}\" \n",
        "\n",
        "p_set = torch.load(PATH_p_set)\n",
        "P_ = torch.load(PATH_P_)#index is the first row\n",
        "print(\"starting from checkpoint\")\n",
        "\n",
        "with open(PATH_i, \"r\") as text_file:\n",
        "  i =int(text_file.read())\n",
        "\n",
        "cardinality = i\n",
        "output = torch.unique(p_set,dim=0)\n",
        "print(\"p_set\",p_set.shape)\n",
        "print(\"output(uniqie row)\",output.shape)"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "starting from checkpoint\n",
            "p_set torch.Size([64, 9])\n",
            "output(uniqie row) torch.Size([64, 9])\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "6YsTIJVAXd28",
        "colab_type": "code",
        "outputId": "4690f664-4a98-4249-e6d2-87794a57876f",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 51
        }
      },
      "source": [
        "\n"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "p_set torch.Size([50, 9])\n",
            "output(uniqie row) torch.Size([50, 9])\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "uBRcEc3EXHMQ",
        "colab_type": "code",
        "outputId": "f22d64c3-870c-41db-e31b-269a47c47d8d",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 68
        }
      },
      "source": [
        "def calculate_avg_hamming(pset):\n",
        "  cardinality=p_set.shape[0]\n",
        "    \n",
        "  D = torch.Tensor().to(device=device)\n",
        "  \n",
        "  sum_=0\n",
        "  for idx,p in enumerate(pset):\n",
        "    P_= torch.cat((pset[:idx,:],pset[idx+1:,:]))\n",
        "    \n",
        "    #hamming distances of ith for all other P'\n",
        "    D_i = torch.stack([(p != p_prime) for p_prime in P_],dim=0).sum(dim=0).view(1,-1).type(torch.FloatTensor).to(device=device)\n",
        "    avg_i = D_i.sum(dim=1)/ (cardinality-1)\n",
        "    sum_=avg_i+sum_\n",
        "    \n",
        "  \n",
        "  return sum_ /cardinality\n",
        "\n",
        "print(\"p_set size x N\")\n",
        "print(p_set.shape)\n",
        "calculate_avg_hamming(p_set)\n",
        "\n"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "p_set size x N\n",
            "torch.Size([100, 9])\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "tensor([8.0800], device='cuda:0')"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 3
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "nyGtkhtXWN0s",
        "colab_type": "code",
        "outputId": "69ce96cd-179b-4f21-ffea-e4de6c35a0ae",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 215
        }
      },
      "source": [
        "\n",
        "      "
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "error",
          "ename": "NameError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-8-955cd1bf39ed>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     28\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     29\u001b[0m \u001b[0;32mwith\u001b[0m \u001b[0mopen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mPATH_i\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m\"w\"\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mtext_file\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 30\u001b[0;31m   \u001b[0mtext_file\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mwrite\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mstr\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mi\u001b[0m\u001b[0;34m+\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     31\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mNameError\u001b[0m: name 'i' is not defined"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "xn_9SgY3ElCG",
        "colab_type": "code",
        "outputId": "152dc89e-11d6-490a-a488-a22f6783cdbb",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "P_ = torch.load(PATH_P_) #index is the first row\n",
        "print(P_.shape)\n",
        "  "
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "torch.Size([9, 362880])\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "1ZiST3swDgJb",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from google.colab import files\n",
        "\n",
        "\n",
        "\n",
        "file_name_p_set = \"checkpoint_p_set.pt.pth\"\n",
        "file_name_P_= \"checkpoint_P_.pt\"\n",
        "file_name_i = \"checkpoint_i.txt\"\n",
        "\n",
        "PATH_p_set = F\"/content/gdrive/My Drive/summerthesis/permutation_set/{file_name_p_set}\" \n",
        "PATH_P_ = F\"/content/gdrive/My Drive/summerthesis/permutation_set/{file_name_P_}\" \n",
        "PATH_i = F\"/content/gdrive/My Drive/summerthesis/permutation_set/{file_name_i}\" \n",
        "\n",
        "#torch.save(model.state_dict(), file_name)\n",
        "#print('saved model to colab')\n",
        "\n",
        "#files.download(file_name)\n",
        "#print('saved model to local pc')\n",
        "\n",
        "torch.save(model.state_dict(), PATH_p_set)\n",
        "torch.save(model.state_dict(), PATH_P_)\n",
        "\n",
        "with open(PATH_i, \"w\") as text_file:\n",
        "      text_file.write(str(p_set.shape[0])\n",
        "                      \n",
        "print('saved  permutation set to google drive')"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "udNrXTa_HsmZ",
        "colab_type": "code",
        "outputId": "73b93876-6fda-4631-a7bd-3341ddb86a69",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 221
        }
      },
      "source": [
        "df = pd.DataFrame(data=np.arange(10))\n",
        "\n"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "   0\n",
            "0  0\n",
            "2  2\n",
            "3  3\n",
            "4  4\n",
            "5  5\n",
            "6  6\n",
            "7  7\n",
            "8  8\n",
            "9  9\n",
            "2\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "['0']"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 39
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "8ppGPqNpOHWH",
        "colab_type": "code",
        "outputId": "cad34195-1953-4b44-ada3-2db1cc335d63",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 85
        }
      },
      "source": [
        "set_p = np.array([0,1,2,3,4,5,6, 7, 8])\n",
        "\n",
        "a = torch.stack([ torch.FloatTensor(p) for p in multiset_permutations(set_p)])\n",
        "print(a.shape)\n",
        "print(a[0])\n",
        "print(a[1])\n",
        "\n",
        "a[1]=a[0]\n",
        "\n",
        "a[2]=a[4]\n",
        "\n",
        "output = torch.unique(a,dim=0)\n",
        "print(output.shape)\n"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "torch.Size([362880, 9])\n",
            "tensor([0., 1., 2., 3., 4., 5., 6., 7., 8.])\n",
            "tensor([0., 1., 2., 3., 4., 5., 6., 8., 7.])\n",
            "torch.Size([362878, 9])\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "h6ciIVU3h5ar",
        "colab_type": "code",
        "outputId": "dcd53465-a01a-4f56-d128-06dbe57347d1",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 51
        }
      },
      "source": [
        "x=torch.tensor([[2., 0., 1., 0., 0., 1., 2., 2., 1., 1., 2., 2., 1., 1., 0., 0., 0., 1., 2., 0., 1., 0., 0.]])\n",
        "\n",
        "print(x.shape)\n",
        "\n",
        "maxx= torch.argmax(x,dim=1)\n",
        "\n",
        "print(maxx)\n"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "torch.Size([1, 23])\n",
            "tensor([18])\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "YCM4ceVwR9rP",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        " \n",
        "  df = pd.read_csv(\"checkpoint_idx.csv\")\n",
        "  df.index = df.iloc[:,0].values\n",
        "  idx = df.drop(['Unnamed: 0'],axis=1)\n",
        "  \n",
        "  with open(\"checkpoint_i.txt\", \"r\") as text_file:\n",
        "    i =int(text_file.read())"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "pE7gPAXwVwoM",
        "colab_type": "code",
        "outputId": "f27eb7bf-923e-47ff-e9ca-16d6a4a8b129",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 238
        }
      },
      "source": [
        "x= torch.zeros(5,5)\n",
        "y= torch.arange(5).type(torch.FloatTensor).view(1,-1)\n",
        "x[:,0]=1\n",
        "x[:,1]=2\n",
        "x[:,3]=3\n",
        "x[:,4]=4\n",
        "x=torch.cat((y,x))\n",
        "print(x)\n",
        "#z=torch.gather(x, 0, torch.tensor([0,0,0,1,0]))\n",
        "indices = (x[0,:] == 2).nonzero()\n",
        "print(indices)\n",
        "\n",
        "z=torch.index_select(x, 1, torch.tensor([0,1,2,3,4]))\n",
        "print(z)"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "tensor([[0., 1., 2., 3., 4.],\n",
            "        [1., 2., 0., 3., 4.],\n",
            "        [1., 2., 0., 3., 4.],\n",
            "        [1., 2., 0., 3., 4.],\n",
            "        [1., 2., 0., 3., 4.],\n",
            "        [1., 2., 0., 3., 4.]])\n",
            "tensor([[2]])\n",
            "tensor([[0., 1., 2., 3., 4.],\n",
            "        [1., 2., 0., 3., 4.],\n",
            "        [1., 2., 0., 3., 4.],\n",
            "        [1., 2., 0., 3., 4.],\n",
            "        [1., 2., 0., 3., 4.],\n",
            "        [1., 2., 0., 3., 4.]])\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "hgwwKnykK_WA",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "##### DENEME delete later\n",
        "set_p = np.array([0,1,2,3,4])\n",
        "\n",
        "P_ = torch.stack([ torch.FloatTensor(p) for p in multiset_permutations(set_p)],dim=1).to(device=device)\n",
        "_,perm_size = P_.shape\n",
        "idx = torch.arange(perm_size).type(torch.FloatTensor).view(1,-1).to(device=device)\n",
        "\n",
        "P_=torch.cat((idx,P_))\n",
        "\n",
        "#print(\"P_(all perm)\", P_.shape)\n",
        "\n",
        "#idx = pd.DataFrame(data=np.arange(perm_size))\n",
        "p_set = torch.Tensor().to(device=device)\n",
        "j=np.random.choice(perm_size)\n",
        "\n",
        "#####"
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}