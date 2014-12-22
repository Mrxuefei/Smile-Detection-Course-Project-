#include <vector>
#include <iostream>
#include "windows.h"
#include <string.h>
#include <Strsafe.h>

using namespace std;

//´«ÈëÒª±éÀúµÄÎÄ¼þ¼ÐÂ·¾¶£¬²¢±éÀúÏàÓ¦ÎÄ¼þ¼Ð
int TraverseDirectory(wchar_t Dir[MAX_PATH], vector<double>& label,vector<string>& paths,double i=1)    
{
	bool empty_file = true;
	WIN32_FIND_DATA FindFileData;
	HANDLE hFind=INVALID_HANDLE_VALUE;
	wchar_t DirSpec[MAX_PATH];                  //¶¨ÒåÒª±éÀúµÄÎÄ¼þ¼ÐµÄÄ¿Â¼
	DWORD dwError;
	StringCchCopy(DirSpec,MAX_PATH,Dir);
    StringCchCat(DirSpec,MAX_PATH,TEXT("\\*"));   //¶¨ÒåÒª±éÀúµÄÎÄ¼þ¼ÐµÄÍêÕûÂ·¾¶\*

	hFind=FindFirstFile(DirSpec,&FindFileData);          //ÕÒµ½ÎÄ¼þ¼ÐÖÐµÄµÚÒ»¸öÎÄ¼þ

	if(hFind==INVALID_HANDLE_VALUE)                               //Èç¹ûhFind¾ä±ú´´½¨Ê§°Ü£¬Êä³ö´íÎóÐÅÏ¢
	{
		FindClose(hFind); 
		return 0;  
	}
	else 
	{
		while(FindNextFile(hFind,&FindFileData)!=0)                            //µ±ÎÄ¼þ»òÕßÎÄ¼þ¼Ð´æÔÚÊ±
		{
			if((FindFileData.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)!=0&&wcscmp(FindFileData.cFileName,L".")==0||wcscmp(FindFileData.cFileName,L"..")==0)        //ÅÐ¶ÏÊÇÎÄ¼þ¼Ð&&±íÊ¾Îª"."||±íÊ¾Îª"."
			{
				 continue;
			}
			if((FindFileData.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)!=0)      //ÅÐ¶ÏÈç¹ûÊÇÎÄ¼þ¼Ð
			{
				wchar_t DirAdd[MAX_PATH];
				StringCchCopy(DirAdd,MAX_PATH,Dir);
				StringCchCat(DirAdd,MAX_PATH,TEXT("\\"));
				StringCchCat(DirAdd,MAX_PATH,FindFileData.cFileName);       //Æ´½ÓµÃµ½´ËÎÄ¼þ¼ÐµÄÍêÕûÂ·¾¶
				i = TraverseDirectory(DirAdd,label,paths,i);                                  //ÊµÏÖµÝ¹éµ÷ÓÃ
			}
			if((FindFileData.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)==0)    //Èç¹û²»ÊÇÎÄ¼þ¼Ð
			{
				empty_file = false;
				wstring ret     = wstring( Dir );
				string p_path(ret.begin(),ret.end());
				wstring _fileName(FindFileData.cFileName);
				string file(_fileName.begin(),_fileName.end());
				if(strstr(file.c_str(),"jpg") != NULL)
				{
					string path = p_path.append("\\").append(file);
					paths.push_back(path);
					label.push_back(i);
					//wcout<<Dir<<"\\"<<FindFileData.cFileName<<"____"<<i<<endl;            //Êä³öÍêÕûÂ·¾¶
				}
			}
		}
		FindClose(hFind);
		if(empty_file)
			return i;
		else
			return ++i;
	}
}
/*
int main( int argc, wchar_t *argv[ ], wchar_t *envp[ ] )
{
	double i=0;
	vector<double> label;
	vector<string> paths;
	TraverseDirectory(L"D:\\food images\\anglos pic\\classfile",label,paths);         //±éÀúÖ¸¶¨µÄÎÄ¼þ¼Ð£¬´Ë´¦ÎÄ¼þÂ·¾¶¿É°´¾ßÌåÇé¿öÐÞ¸Ä
	for(int i=0;i<paths.size();i++)
		cout<<paths.at(i).c_str()<<endl;
	system("pause");
	
	return 0;
}*/
