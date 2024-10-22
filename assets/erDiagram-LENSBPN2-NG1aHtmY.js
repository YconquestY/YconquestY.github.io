import{s as Mt}from"./chunk-BOP2KBYH-Dk9ufznx.js";import{b as wt}from"./chunk-6XGRHI2A-sfbOt74q.js";import{m as s,J as U,v as vt,Q as W,w as St,B as $t,F as Dt,M as It,L as Lt,A as Bt,h as Ct,i as Pt,j as Yt,b as at,I as Zt,k as Ft}from"./mermaid.esm.min-cEHSL7td.js";import"./chunk-BKDDFIKN-DIMbnRM6.js";import"./app-DhNDvZEA.js";var nt=function(){var t=s(function(v,a,o,n){for(o=o||{},n=v.length;n--;o[v[n]]=a);return o},"o"),e=[6,8,10,20,22,24,26,27,28],r=[1,10],y=[1,11],h=[1,12],_=[1,13],d=[1,14],c=[1,15],u=[1,21],g=[1,22],E=[1,23],m=[1,24],k=[1,25],f=[6,8,10,13,15,18,19,20,22,24,26,27,28,41,42,43,44,45],x=[1,34],w=[27,28,46,47],P=[41,42,43,44,45],Y=[17,34],Z=[1,54],T=[1,53],S=[17,34,36,38],O={trace:s(function(){},"trace"),yy:{},symbols_:{error:2,start:3,ER_DIAGRAM:4,document:5,EOF:6,line:7,SPACE:8,statement:9,NEWLINE:10,entityName:11,relSpec:12,":":13,role:14,BLOCK_START:15,attributes:16,BLOCK_STOP:17,SQS:18,SQE:19,title:20,title_value:21,acc_title:22,acc_title_value:23,acc_descr:24,acc_descr_value:25,acc_descr_multiline_value:26,ALPHANUM:27,ENTITY_NAME:28,attribute:29,attributeType:30,attributeName:31,attributeKeyTypeList:32,attributeComment:33,ATTRIBUTE_WORD:34,attributeKeyType:35,COMMA:36,ATTRIBUTE_KEY:37,COMMENT:38,cardinality:39,relType:40,ZERO_OR_ONE:41,ZERO_OR_MORE:42,ONE_OR_MORE:43,ONLY_ONE:44,MD_PARENT:45,NON_IDENTIFYING:46,IDENTIFYING:47,WORD:48,$accept:0,$end:1},terminals_:{2:"error",4:"ER_DIAGRAM",6:"EOF",8:"SPACE",10:"NEWLINE",13:":",15:"BLOCK_START",17:"BLOCK_STOP",18:"SQS",19:"SQE",20:"title",21:"title_value",22:"acc_title",23:"acc_title_value",24:"acc_descr",25:"acc_descr_value",26:"acc_descr_multiline_value",27:"ALPHANUM",28:"ENTITY_NAME",34:"ATTRIBUTE_WORD",36:"COMMA",37:"ATTRIBUTE_KEY",38:"COMMENT",41:"ZERO_OR_ONE",42:"ZERO_OR_MORE",43:"ONE_OR_MORE",44:"ONLY_ONE",45:"MD_PARENT",46:"NON_IDENTIFYING",47:"IDENTIFYING",48:"WORD"},productions_:[0,[3,3],[5,0],[5,2],[7,2],[7,1],[7,1],[7,1],[9,5],[9,4],[9,3],[9,1],[9,7],[9,6],[9,4],[9,2],[9,2],[9,2],[9,1],[11,1],[11,1],[16,1],[16,2],[29,2],[29,3],[29,3],[29,4],[30,1],[31,1],[32,1],[32,3],[35,1],[33,1],[12,3],[39,1],[39,1],[39,1],[39,1],[39,1],[40,1],[40,1],[14,1],[14,1],[14,1]],performAction:s(function(v,a,o,n,p,i,A){var l=i.length-1;switch(p){case 1:break;case 2:this.$=[];break;case 3:i[l-1].push(i[l]),this.$=i[l-1];break;case 4:case 5:this.$=i[l];break;case 6:case 7:this.$=[];break;case 8:n.addEntity(i[l-4]),n.addEntity(i[l-2]),n.addRelationship(i[l-4],i[l],i[l-2],i[l-3]);break;case 9:n.addEntity(i[l-3]),n.addAttributes(i[l-3],i[l-1]);break;case 10:n.addEntity(i[l-2]);break;case 11:n.addEntity(i[l]);break;case 12:n.addEntity(i[l-6],i[l-4]),n.addAttributes(i[l-6],i[l-1]);break;case 13:n.addEntity(i[l-5],i[l-3]);break;case 14:n.addEntity(i[l-3],i[l-1]);break;case 15:case 16:this.$=i[l].trim(),n.setAccTitle(this.$);break;case 17:case 18:this.$=i[l].trim(),n.setAccDescription(this.$);break;case 19:case 43:this.$=i[l];break;case 20:case 41:case 42:this.$=i[l].replace(/"/g,"");break;case 21:case 29:this.$=[i[l]];break;case 22:i[l].push(i[l-1]),this.$=i[l];break;case 23:this.$={attributeType:i[l-1],attributeName:i[l]};break;case 24:this.$={attributeType:i[l-2],attributeName:i[l-1],attributeKeyTypeList:i[l]};break;case 25:this.$={attributeType:i[l-2],attributeName:i[l-1],attributeComment:i[l]};break;case 26:this.$={attributeType:i[l-3],attributeName:i[l-2],attributeKeyTypeList:i[l-1],attributeComment:i[l]};break;case 27:case 28:case 31:this.$=i[l];break;case 30:i[l-2].push(i[l]),this.$=i[l-2];break;case 32:this.$=i[l].replace(/"/g,"");break;case 33:this.$={cardA:i[l],relType:i[l-1],cardB:i[l-2]};break;case 34:this.$=n.Cardinality.ZERO_OR_ONE;break;case 35:this.$=n.Cardinality.ZERO_OR_MORE;break;case 36:this.$=n.Cardinality.ONE_OR_MORE;break;case 37:this.$=n.Cardinality.ONLY_ONE;break;case 38:this.$=n.Cardinality.MD_PARENT;break;case 39:this.$=n.Identification.NON_IDENTIFYING;break;case 40:this.$=n.Identification.IDENTIFYING;break}},"anonymous"),table:[{3:1,4:[1,2]},{1:[3]},t(e,[2,2],{5:3}),{6:[1,4],7:5,8:[1,6],9:7,10:[1,8],11:9,20:r,22:y,24:h,26:_,27:d,28:c},t(e,[2,7],{1:[2,1]}),t(e,[2,3]),{9:16,11:9,20:r,22:y,24:h,26:_,27:d,28:c},t(e,[2,5]),t(e,[2,6]),t(e,[2,11],{12:17,39:20,15:[1,18],18:[1,19],41:u,42:g,43:E,44:m,45:k}),{21:[1,26]},{23:[1,27]},{25:[1,28]},t(e,[2,18]),t(f,[2,19]),t(f,[2,20]),t(e,[2,4]),{11:29,27:d,28:c},{16:30,17:[1,31],29:32,30:33,34:x},{11:35,27:d,28:c},{40:36,46:[1,37],47:[1,38]},t(w,[2,34]),t(w,[2,35]),t(w,[2,36]),t(w,[2,37]),t(w,[2,38]),t(e,[2,15]),t(e,[2,16]),t(e,[2,17]),{13:[1,39]},{17:[1,40]},t(e,[2,10]),{16:41,17:[2,21],29:32,30:33,34:x},{31:42,34:[1,43]},{34:[2,27]},{19:[1,44]},{39:45,41:u,42:g,43:E,44:m,45:k},t(P,[2,39]),t(P,[2,40]),{14:46,27:[1,49],28:[1,48],48:[1,47]},t(e,[2,9]),{17:[2,22]},t(Y,[2,23],{32:50,33:51,35:52,37:Z,38:T}),t([17,34,37,38],[2,28]),t(e,[2,14],{15:[1,55]}),t([27,28],[2,33]),t(e,[2,8]),t(e,[2,41]),t(e,[2,42]),t(e,[2,43]),t(Y,[2,24],{33:56,36:[1,57],38:T}),t(Y,[2,25]),t(S,[2,29]),t(Y,[2,32]),t(S,[2,31]),{16:58,17:[1,59],29:32,30:33,34:x},t(Y,[2,26]),{35:60,37:Z},{17:[1,61]},t(e,[2,13]),t(S,[2,30]),t(e,[2,12])],defaultActions:{34:[2,27],41:[2,22]},parseError:s(function(v,a){if(a.recoverable)this.trace(v);else{var o=new Error(v);throw o.hash=a,o}},"parseError"),parse:s(function(v){var a=this,o=[0],n=[],p=[null],i=[],A=this.table,l="",j=0,lt=0,Rt=0,Nt=2,ct=1,xt=i.slice.call(arguments,1),N=Object.create(this.lexer),K={yy:{}};for(var J in this.yy)Object.prototype.hasOwnProperty.call(this.yy,J)&&(K.yy[J]=this.yy[J]);N.setInput(v,K.yy),K.yy.lexer=N,K.yy.parser=this,typeof N.yylloc>"u"&&(N.yylloc={});var tt=N.yylloc;i.push(tt);var Tt=N.options&&N.options.ranges;typeof K.yy.parseError=="function"?this.parseError=K.yy.parseError:this.parseError=Object.getPrototypeOf(this).parseError;function At(D){o.length=o.length-2*D,p.length=p.length-D,i.length=i.length-D}s(At,"popStack");function ht(){var D;return D=n.pop()||N.lex()||ct,typeof D!="number"&&(D instanceof Array&&(n=D,D=n.pop()),D=a.symbols_[D]||D),D}s(ht,"lex");for(var $,et,z,L,ge,rt,Q={},G,F,dt,q;;){if(z=o[o.length-1],this.defaultActions[z]?L=this.defaultActions[z]:(($===null||typeof $>"u")&&($=ht()),L=A[z]&&A[z][$]),typeof L>"u"||!L.length||!L[0]){var it="";q=[];for(G in A[z])this.terminals_[G]&&G>Nt&&q.push("'"+this.terminals_[G]+"'");N.showPosition?it="Parse error on line "+(j+1)+`:
`+N.showPosition()+`
Expecting `+q.join(", ")+", got '"+(this.terminals_[$]||$)+"'":it="Parse error on line "+(j+1)+": Unexpected "+($==ct?"end of input":"'"+(this.terminals_[$]||$)+"'"),this.parseError(it,{text:N.match,token:this.terminals_[$]||$,line:N.yylineno,loc:tt,expected:q})}if(L[0]instanceof Array&&L.length>1)throw new Error("Parse Error: multiple actions possible at state: "+z+", token: "+$);switch(L[0]){case 1:o.push($),p.push(N.yytext),i.push(N.yylloc),o.push(L[1]),$=null,et?($=et,et=null):(lt=N.yyleng,l=N.yytext,j=N.yylineno,tt=N.yylloc,Rt>0);break;case 2:if(F=this.productions_[L[1]][1],Q.$=p[p.length-F],Q._$={first_line:i[i.length-(F||1)].first_line,last_line:i[i.length-1].last_line,first_column:i[i.length-(F||1)].first_column,last_column:i[i.length-1].last_column},Tt&&(Q._$.range=[i[i.length-(F||1)].range[0],i[i.length-1].range[1]]),rt=this.performAction.apply(Q,[l,lt,j,K.yy,L[1],p,i].concat(xt)),typeof rt<"u")return rt;F&&(o=o.slice(0,-1*F*2),p=p.slice(0,-1*F),i=i.slice(0,-1*F)),o.push(this.productions_[L[1]][0]),p.push(Q.$),i.push(Q._$),dt=A[o[o.length-2]][o[o.length-1]],o.push(dt);break;case 3:return!0}}return!0},"parse")},R=function(){var v={EOF:1,parseError:s(function(a,o){if(this.yy.parser)this.yy.parser.parseError(a,o);else throw new Error(a)},"parseError"),setInput:s(function(a,o){return this.yy=o||this.yy||{},this._input=a,this._more=this._backtrack=this.done=!1,this.yylineno=this.yyleng=0,this.yytext=this.matched=this.match="",this.conditionStack=["INITIAL"],this.yylloc={first_line:1,first_column:0,last_line:1,last_column:0},this.options.ranges&&(this.yylloc.range=[0,0]),this.offset=0,this},"setInput"),input:s(function(){var a=this._input[0];this.yytext+=a,this.yyleng++,this.offset++,this.match+=a,this.matched+=a;var o=a.match(/(?:\r\n?|\n).*/g);return o?(this.yylineno++,this.yylloc.last_line++):this.yylloc.last_column++,this.options.ranges&&this.yylloc.range[1]++,this._input=this._input.slice(1),a},"input"),unput:s(function(a){var o=a.length,n=a.split(/(?:\r\n?|\n)/g);this._input=a+this._input,this.yytext=this.yytext.substr(0,this.yytext.length-o),this.offset-=o;var p=this.match.split(/(?:\r\n?|\n)/g);this.match=this.match.substr(0,this.match.length-1),this.matched=this.matched.substr(0,this.matched.length-1),n.length-1&&(this.yylineno-=n.length-1);var i=this.yylloc.range;return this.yylloc={first_line:this.yylloc.first_line,last_line:this.yylineno+1,first_column:this.yylloc.first_column,last_column:n?(n.length===p.length?this.yylloc.first_column:0)+p[p.length-n.length].length-n[0].length:this.yylloc.first_column-o},this.options.ranges&&(this.yylloc.range=[i[0],i[0]+this.yyleng-o]),this.yyleng=this.yytext.length,this},"unput"),more:s(function(){return this._more=!0,this},"more"),reject:s(function(){if(this.options.backtrack_lexer)this._backtrack=!0;else return this.parseError("Lexical error on line "+(this.yylineno+1)+`. You can only invoke reject() in the lexer when the lexer is of the backtracking persuasion (options.backtrack_lexer = true).
`+this.showPosition(),{text:"",token:null,line:this.yylineno});return this},"reject"),less:s(function(a){this.unput(this.match.slice(a))},"less"),pastInput:s(function(){var a=this.matched.substr(0,this.matched.length-this.match.length);return(a.length>20?"...":"")+a.substr(-20).replace(/\n/g,"")},"pastInput"),upcomingInput:s(function(){var a=this.match;return a.length<20&&(a+=this._input.substr(0,20-a.length)),(a.substr(0,20)+(a.length>20?"...":"")).replace(/\n/g,"")},"upcomingInput"),showPosition:s(function(){var a=this.pastInput(),o=new Array(a.length+1).join("-");return a+this.upcomingInput()+`
`+o+"^"},"showPosition"),test_match:s(function(a,o){var n,p,i;if(this.options.backtrack_lexer&&(i={yylineno:this.yylineno,yylloc:{first_line:this.yylloc.first_line,last_line:this.last_line,first_column:this.yylloc.first_column,last_column:this.yylloc.last_column},yytext:this.yytext,match:this.match,matches:this.matches,matched:this.matched,yyleng:this.yyleng,offset:this.offset,_more:this._more,_input:this._input,yy:this.yy,conditionStack:this.conditionStack.slice(0),done:this.done},this.options.ranges&&(i.yylloc.range=this.yylloc.range.slice(0))),p=a[0].match(/(?:\r\n?|\n).*/g),p&&(this.yylineno+=p.length),this.yylloc={first_line:this.yylloc.last_line,last_line:this.yylineno+1,first_column:this.yylloc.last_column,last_column:p?p[p.length-1].length-p[p.length-1].match(/\r?\n?/)[0].length:this.yylloc.last_column+a[0].length},this.yytext+=a[0],this.match+=a[0],this.matches=a,this.yyleng=this.yytext.length,this.options.ranges&&(this.yylloc.range=[this.offset,this.offset+=this.yyleng]),this._more=!1,this._backtrack=!1,this._input=this._input.slice(a[0].length),this.matched+=a[0],n=this.performAction.call(this,this.yy,this,o,this.conditionStack[this.conditionStack.length-1]),this.done&&this._input&&(this.done=!1),n)return n;if(this._backtrack){for(var A in i)this[A]=i[A];return!1}return!1},"test_match"),next:s(function(){if(this.done)return this.EOF;this._input||(this.done=!0);var a,o,n,p;this._more||(this.yytext="",this.match="");for(var i=this._currentRules(),A=0;A<i.length;A++)if(n=this._input.match(this.rules[i[A]]),n&&(!o||n[0].length>o[0].length)){if(o=n,p=A,this.options.backtrack_lexer){if(a=this.test_match(n,i[A]),a!==!1)return a;if(this._backtrack){o=!1;continue}else return!1}else if(!this.options.flex)break}return o?(a=this.test_match(o,i[p]),a!==!1?a:!1):this._input===""?this.EOF:this.parseError("Lexical error on line "+(this.yylineno+1)+`. Unrecognized text.
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},"next"),lex:s(function(){var a=this.next();return a||this.lex()},"lex"),begin:s(function(a){this.conditionStack.push(a)},"begin"),popState:s(function(){var a=this.conditionStack.length-1;return a>0?this.conditionStack.pop():this.conditionStack[0]},"popState"),_currentRules:s(function(){return this.conditionStack.length&&this.conditionStack[this.conditionStack.length-1]?this.conditions[this.conditionStack[this.conditionStack.length-1]].rules:this.conditions.INITIAL.rules},"_currentRules"),topState:s(function(a){return a=this.conditionStack.length-1-Math.abs(a||0),a>=0?this.conditionStack[a]:"INITIAL"},"topState"),pushState:s(function(a){this.begin(a)},"pushState"),stateStackSize:s(function(){return this.conditionStack.length},"stateStackSize"),options:{"case-insensitive":!0},performAction:s(function(a,o,n,p){switch(n){case 0:return this.begin("acc_title"),22;case 1:return this.popState(),"acc_title_value";case 2:return this.begin("acc_descr"),24;case 3:return this.popState(),"acc_descr_value";case 4:this.begin("acc_descr_multiline");break;case 5:this.popState();break;case 6:return"acc_descr_multiline_value";case 7:return 10;case 8:break;case 9:return 8;case 10:return 28;case 11:return 48;case 12:return 4;case 13:return this.begin("block"),15;case 14:return 36;case 15:break;case 16:return 37;case 17:return 34;case 18:return 34;case 19:return 38;case 20:break;case 21:return this.popState(),17;case 22:return o.yytext[0];case 23:return 18;case 24:return 19;case 25:return 41;case 26:return 43;case 27:return 43;case 28:return 43;case 29:return 41;case 30:return 41;case 31:return 42;case 32:return 42;case 33:return 42;case 34:return 42;case 35:return 42;case 36:return 43;case 37:return 42;case 38:return 43;case 39:return 44;case 40:return 44;case 41:return 44;case 42:return 44;case 43:return 41;case 44:return 42;case 45:return 43;case 46:return 45;case 47:return 46;case 48:return 47;case 49:return 47;case 50:return 46;case 51:return 46;case 52:return 46;case 53:return 27;case 54:return o.yytext[0];case 55:return 6}},"anonymous"),rules:[/^(?:accTitle\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*\{\s*)/i,/^(?:[\}])/i,/^(?:[^\}]*)/i,/^(?:[\n]+)/i,/^(?:\s+)/i,/^(?:[\s]+)/i,/^(?:"[^"%\r\n\v\b\\]+")/i,/^(?:"[^"]*")/i,/^(?:erDiagram\b)/i,/^(?:\{)/i,/^(?:,)/i,/^(?:\s+)/i,/^(?:\b((?:PK)|(?:FK)|(?:UK))\b)/i,/^(?:(.*?)[~](.*?)*[~])/i,/^(?:[\*A-Za-z_][A-Za-z0-9\-_\[\]\(\)]*)/i,/^(?:"[^"]*")/i,/^(?:[\n]+)/i,/^(?:\})/i,/^(?:.)/i,/^(?:\[)/i,/^(?:\])/i,/^(?:one or zero\b)/i,/^(?:one or more\b)/i,/^(?:one or many\b)/i,/^(?:1\+)/i,/^(?:\|o\b)/i,/^(?:zero or one\b)/i,/^(?:zero or more\b)/i,/^(?:zero or many\b)/i,/^(?:0\+)/i,/^(?:\}o\b)/i,/^(?:many\(0\))/i,/^(?:many\(1\))/i,/^(?:many\b)/i,/^(?:\}\|)/i,/^(?:one\b)/i,/^(?:only one\b)/i,/^(?:1\b)/i,/^(?:\|\|)/i,/^(?:o\|)/i,/^(?:o\{)/i,/^(?:\|\{)/i,/^(?:\s*u\b)/i,/^(?:\.\.)/i,/^(?:--)/i,/^(?:to\b)/i,/^(?:optionally to\b)/i,/^(?:\.-)/i,/^(?:-\.)/i,/^(?:[A-Za-z_][A-Za-z0-9\-_]*)/i,/^(?:.)/i,/^(?:$)/i],conditions:{acc_descr_multiline:{rules:[5,6],inclusive:!1},acc_descr:{rules:[3],inclusive:!1},acc_title:{rules:[1],inclusive:!1},block:{rules:[14,15,16,17,18,19,20,21,22],inclusive:!1},INITIAL:{rules:[0,2,4,7,8,9,10,11,12,13,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55],inclusive:!0}}};return v}();O.lexer=R;function I(){this.yy={}}return s(I,"Parser"),I.prototype=O,O.Parser=I,new I}();nt.parser=nt;var Wt=nt,H=new Map,ot=[],Kt={ZERO_OR_ONE:"ZERO_OR_ONE",ZERO_OR_MORE:"ZERO_OR_MORE",ONE_OR_MORE:"ONE_OR_MORE",ONLY_ONE:"ONLY_ONE",MD_PARENT:"MD_PARENT"},zt={NON_IDENTIFYING:"NON_IDENTIFYING",IDENTIFYING:"IDENTIFYING"},yt=s(function(t,e=void 0){return H.has(t)?!H.get(t).alias&&e&&(H.get(t).alias=e,U.info(`Add alias '${e}' to entity '${t}'`)):(H.set(t,{attributes:[],alias:e}),U.info("Added new entity :",t)),H.get(t)},"addEntity"),Ht=s(()=>H,"getEntities"),Qt=s(function(t,e){let r=yt(t),y;for(y=e.length-1;y>=0;y--)r.attributes.push(e[y]),U.debug("Added attribute ",e[y].attributeName)},"addAttributes"),Ut=s(function(t,e,r,y){let h={entityA:t,roleA:e,entityB:r,relSpec:y};ot.push(h),U.debug("Added new relationship :",h)},"addRelationship"),Xt=s(()=>ot,"getRelationships"),jt=s(function(){H=new Map,ot=[],vt()},"clear"),Gt={Cardinality:Kt,Identification:zt,getConfig:s(()=>W().er,"getConfig"),addEntity:yt,addAttributes:Qt,getEntities:Ht,addRelationship:Ut,getRelationships:Xt,clear:jt,setAccTitle:St,getAccTitle:$t,setAccDescription:Dt,getAccDescription:It,setDiagramTitle:Lt,getDiagramTitle:Bt},B={ONLY_ONE_START:"ONLY_ONE_START",ONLY_ONE_END:"ONLY_ONE_END",ZERO_OR_ONE_START:"ZERO_OR_ONE_START",ZERO_OR_ONE_END:"ZERO_OR_ONE_END",ONE_OR_MORE_START:"ONE_OR_MORE_START",ONE_OR_MORE_END:"ONE_OR_MORE_END",ZERO_OR_MORE_START:"ZERO_OR_MORE_START",ZERO_OR_MORE_END:"ZERO_OR_MORE_END",MD_PARENT_END:"MD_PARENT_END",MD_PARENT_START:"MD_PARENT_START"},qt=s(function(t,e){let r;t.append("defs").append("marker").attr("id",B.MD_PARENT_START).attr("refX",0).attr("refY",7).attr("markerWidth",190).attr("markerHeight",240).attr("orient","auto").append("path").attr("d","M 18,7 L9,13 L1,7 L9,1 Z"),t.append("defs").append("marker").attr("id",B.MD_PARENT_END).attr("refX",19).attr("refY",7).attr("markerWidth",20).attr("markerHeight",28).attr("orient","auto").append("path").attr("d","M 18,7 L9,13 L1,7 L9,1 Z"),t.append("defs").append("marker").attr("id",B.ONLY_ONE_START).attr("refX",0).attr("refY",9).attr("markerWidth",18).attr("markerHeight",18).attr("orient","auto").append("path").attr("stroke",e.stroke).attr("fill","none").attr("d","M9,0 L9,18 M15,0 L15,18"),t.append("defs").append("marker").attr("id",B.ONLY_ONE_END).attr("refX",18).attr("refY",9).attr("markerWidth",18).attr("markerHeight",18).attr("orient","auto").append("path").attr("stroke",e.stroke).attr("fill","none").attr("d","M3,0 L3,18 M9,0 L9,18"),r=t.append("defs").append("marker").attr("id",B.ZERO_OR_ONE_START).attr("refX",0).attr("refY",9).attr("markerWidth",30).attr("markerHeight",18).attr("orient","auto"),r.append("circle").attr("stroke",e.stroke).attr("fill","white").attr("cx",21).attr("cy",9).attr("r",6),r.append("path").attr("stroke",e.stroke).attr("fill","none").attr("d","M9,0 L9,18"),r=t.append("defs").append("marker").attr("id",B.ZERO_OR_ONE_END).attr("refX",30).attr("refY",9).attr("markerWidth",30).attr("markerHeight",18).attr("orient","auto"),r.append("circle").attr("stroke",e.stroke).attr("fill","white").attr("cx",9).attr("cy",9).attr("r",6),r.append("path").attr("stroke",e.stroke).attr("fill","none").attr("d","M21,0 L21,18"),t.append("defs").append("marker").attr("id",B.ONE_OR_MORE_START).attr("refX",18).attr("refY",18).attr("markerWidth",45).attr("markerHeight",36).attr("orient","auto").append("path").attr("stroke",e.stroke).attr("fill","none").attr("d","M0,18 Q 18,0 36,18 Q 18,36 0,18 M42,9 L42,27"),t.append("defs").append("marker").attr("id",B.ONE_OR_MORE_END).attr("refX",27).attr("refY",18).attr("markerWidth",45).attr("markerHeight",36).attr("orient","auto").append("path").attr("stroke",e.stroke).attr("fill","none").attr("d","M3,9 L3,27 M9,18 Q27,0 45,18 Q27,36 9,18"),r=t.append("defs").append("marker").attr("id",B.ZERO_OR_MORE_START).attr("refX",18).attr("refY",18).attr("markerWidth",57).attr("markerHeight",36).attr("orient","auto"),r.append("circle").attr("stroke",e.stroke).attr("fill","white").attr("cx",48).attr("cy",18).attr("r",6),r.append("path").attr("stroke",e.stroke).attr("fill","none").attr("d","M0,18 Q18,0 36,18 Q18,36 0,18"),r=t.append("defs").append("marker").attr("id",B.ZERO_OR_MORE_END).attr("refX",39).attr("refY",18).attr("markerWidth",57).attr("markerHeight",36).attr("orient","auto"),r.append("circle").attr("stroke",e.stroke).attr("fill","white").attr("cx",9).attr("cy",18).attr("r",6),r.append("path").attr("stroke",e.stroke).attr("fill","none").attr("d","M21,18 Q39,0 57,18 Q39,36 21,18")},"insertMarkers"),C={ERMarkers:B,insertMarkers:qt},Vt=/^(?:[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}|00000000-0000-0000-0000-000000000000)$/i;function pt(t){return typeof t=="string"&&Vt.test(t)}s(pt,"validate");var Jt=pt,M=[];for(let t=0;t<256;++t)M.push((t+256).toString(16).slice(1));function _t(t,e=0){return M[t[e+0]]+M[t[e+1]]+M[t[e+2]]+M[t[e+3]]+"-"+M[t[e+4]]+M[t[e+5]]+"-"+M[t[e+6]]+M[t[e+7]]+"-"+M[t[e+8]]+M[t[e+9]]+"-"+M[t[e+10]]+M[t[e+11]]+M[t[e+12]]+M[t[e+13]]+M[t[e+14]]+M[t[e+15]]}s(_t,"unsafeStringify");function ft(t){if(!Jt(t))throw TypeError("Invalid UUID");let e,r=new Uint8Array(16);return r[0]=(e=parseInt(t.slice(0,8),16))>>>24,r[1]=e>>>16&255,r[2]=e>>>8&255,r[3]=e&255,r[4]=(e=parseInt(t.slice(9,13),16))>>>8,r[5]=e&255,r[6]=(e=parseInt(t.slice(14,18),16))>>>8,r[7]=e&255,r[8]=(e=parseInt(t.slice(19,23),16))>>>8,r[9]=e&255,r[10]=(e=parseInt(t.slice(24,36),16))/1099511627776&255,r[11]=e/4294967296&255,r[12]=e>>>24&255,r[13]=e>>>16&255,r[14]=e>>>8&255,r[15]=e&255,r}s(ft,"parse");var te=ft;function Et(t){t=unescape(encodeURIComponent(t));let e=[];for(let r=0;r<t.length;++r)e.push(t.charCodeAt(r));return e}s(Et,"stringToBytes");var ee="6ba7b810-9dad-11d1-80b4-00c04fd430c8",re="6ba7b811-9dad-11d1-80b4-00c04fd430c8";function gt(t,e,r){function y(h,_,d,c){var u;if(typeof h=="string"&&(h=Et(h)),typeof _=="string"&&(_=te(_)),((u=_)===null||u===void 0?void 0:u.length)!==16)throw TypeError("Namespace must be array-like (16 iterable integer values, 0-255)");let g=new Uint8Array(16+h.length);if(g.set(_),g.set(h,_.length),g=r(g),g[6]=g[6]&15|e,g[8]=g[8]&63|128,d){c=c||0;for(let E=0;E<16;++E)d[c+E]=g[E];return d}return _t(g)}s(y,"generateUUID");try{y.name=t}catch{}return y.DNS=ee,y.URL=re,y}s(gt,"v35");function mt(t,e,r,y){switch(t){case 0:return e&r^~e&y;case 1:return e^r^y;case 2:return e&r^e&y^r&y;case 3:return e^r^y}}s(mt,"f");function V(t,e){return t<<e|t>>>32-e}s(V,"ROTL");function kt(t){let e=[1518500249,1859775393,2400959708,3395469782],r=[1732584193,4023233417,2562383102,271733878,3285377520];if(typeof t=="string"){let d=unescape(encodeURIComponent(t));t=[];for(let c=0;c<d.length;++c)t.push(d.charCodeAt(c))}else Array.isArray(t)||(t=Array.prototype.slice.call(t));t.push(128);let y=t.length/4+2,h=Math.ceil(y/16),_=new Array(h);for(let d=0;d<h;++d){let c=new Uint32Array(16);for(let u=0;u<16;++u)c[u]=t[d*64+u*4]<<24|t[d*64+u*4+1]<<16|t[d*64+u*4+2]<<8|t[d*64+u*4+3];_[d]=c}_[h-1][14]=(t.length-1)*8/Math.pow(2,32),_[h-1][14]=Math.floor(_[h-1][14]),_[h-1][15]=(t.length-1)*8&4294967295;for(let d=0;d<h;++d){let c=new Uint32Array(80);for(let f=0;f<16;++f)c[f]=_[d][f];for(let f=16;f<80;++f)c[f]=V(c[f-3]^c[f-8]^c[f-14]^c[f-16],1);let u=r[0],g=r[1],E=r[2],m=r[3],k=r[4];for(let f=0;f<80;++f){let x=Math.floor(f/20),w=V(u,5)+mt(x,g,E,m)+k+e[x]+c[f]>>>0;k=m,m=E,E=V(g,30)>>>0,g=u,u=w}r[0]=r[0]+u>>>0,r[1]=r[1]+g>>>0,r[2]=r[2]+E>>>0,r[3]=r[3]+m>>>0,r[4]=r[4]+k>>>0}return[r[0]>>24&255,r[0]>>16&255,r[0]>>8&255,r[0]&255,r[1]>>24&255,r[1]>>16&255,r[1]>>8&255,r[1]&255,r[2]>>24&255,r[2]>>16&255,r[2]>>8&255,r[2]&255,r[3]>>24&255,r[3]>>16&255,r[3]>>8&255,r[3]&255,r[4]>>24&255,r[4]>>16&255,r[4]>>8&255,r[4]&255]}s(kt,"sha1");var ie=kt,ae=gt("v5",80,ie),ne=ae,se=/[^\dA-Za-z](\W)*/g,b={},X=new Map,oe=s(function(t){let e=Object.keys(t);for(let r of e)b[r]=t[r]},"setConf"),le=s((t,e,r)=>{let y=b.entityPadding/3,h=b.entityPadding/3,_=b.fontSize*.85,d=e.node().getBBox(),c=[],u=!1,g=!1,E=0,m=0,k=0,f=0,x=d.height+y*2,w=1;r.forEach(T=>{T.attributeKeyTypeList!==void 0&&T.attributeKeyTypeList.length>0&&(u=!0),T.attributeComment!==void 0&&(g=!0)}),r.forEach(T=>{let S=`${e.node().id}-attr-${w}`,O=0,R=Ct(T.attributeType),I=t.append("text").classed("er entityLabel",!0).attr("id",`${S}-type`).attr("x",0).attr("y",0).style("dominant-baseline","middle").style("text-anchor","left").style("font-family",W().fontFamily).style("font-size",_+"px").text(R),v=t.append("text").classed("er entityLabel",!0).attr("id",`${S}-name`).attr("x",0).attr("y",0).style("dominant-baseline","middle").style("text-anchor","left").style("font-family",W().fontFamily).style("font-size",_+"px").text(T.attributeName),a={};a.tn=I,a.nn=v;let o=I.node().getBBox(),n=v.node().getBBox();if(E=Math.max(E,o.width),m=Math.max(m,n.width),O=Math.max(o.height,n.height),u){let p=T.attributeKeyTypeList!==void 0?T.attributeKeyTypeList.join(","):"",i=t.append("text").classed("er entityLabel",!0).attr("id",`${S}-key`).attr("x",0).attr("y",0).style("dominant-baseline","middle").style("text-anchor","left").style("font-family",W().fontFamily).style("font-size",_+"px").text(p);a.kn=i;let A=i.node().getBBox();k=Math.max(k,A.width),O=Math.max(O,A.height)}if(g){let p=t.append("text").classed("er entityLabel",!0).attr("id",`${S}-comment`).attr("x",0).attr("y",0).style("dominant-baseline","middle").style("text-anchor","left").style("font-family",W().fontFamily).style("font-size",_+"px").text(T.attributeComment||"");a.cn=p;let i=p.node().getBBox();f=Math.max(f,i.width),O=Math.max(O,i.height)}a.height=O,c.push(a),x+=O+y*2,w+=1});let P=4;u&&(P+=2),g&&(P+=2);let Y=E+m+k+f,Z={width:Math.max(b.minEntityWidth,Math.max(d.width+b.entityPadding*2,Y+h*P)),height:r.length>0?x:Math.max(b.minEntityHeight,d.height+b.entityPadding*2)};if(r.length>0){let T=Math.max(0,(Z.width-Y-h*P)/(P/2));e.attr("transform","translate("+Z.width/2+","+(y+d.height/2)+")");let S=d.height+y*2,O="attributeBoxOdd";c.forEach(R=>{let I=S+y+R.height/2;R.tn.attr("transform","translate("+h+","+I+")");let v=t.insert("rect","#"+R.tn.node().id).classed(`er ${O}`,!0).attr("x",0).attr("y",S).attr("width",E+h*2+T).attr("height",R.height+y*2),a=parseFloat(v.attr("x"))+parseFloat(v.attr("width"));R.nn.attr("transform","translate("+(a+h)+","+I+")");let o=t.insert("rect","#"+R.nn.node().id).classed(`er ${O}`,!0).attr("x",a).attr("y",S).attr("width",m+h*2+T).attr("height",R.height+y*2),n=parseFloat(o.attr("x"))+parseFloat(o.attr("width"));if(u){R.kn.attr("transform","translate("+(n+h)+","+I+")");let p=t.insert("rect","#"+R.kn.node().id).classed(`er ${O}`,!0).attr("x",n).attr("y",S).attr("width",k+h*2+T).attr("height",R.height+y*2);n=parseFloat(p.attr("x"))+parseFloat(p.attr("width"))}g&&(R.cn.attr("transform","translate("+(n+h)+","+I+")"),t.insert("rect","#"+R.cn.node().id).classed(`er ${O}`,"true").attr("x",n).attr("y",S).attr("width",f+h*2+T).attr("height",R.height+y*2)),S+=R.height+y*2,O=O==="attributeBoxOdd"?"attributeBoxEven":"attributeBoxOdd"})}else Z.height=Math.max(b.minEntityHeight,x),e.attr("transform","translate("+Z.width/2+","+Z.height/2+")");return Z},"drawAttributes"),ce=s(function(t,e,r){let y=[...e.keys()],h;return y.forEach(function(_){let d=Ot(_,"entity");X.set(_,d);let c=t.append("g").attr("id",d);h=h===void 0?d:h;let u="text-"+d,g=c.append("text").classed("er entityLabel",!0).attr("id",u).attr("x",0).attr("y",0).style("dominant-baseline","middle").style("text-anchor","middle").style("font-family",W().fontFamily).style("font-size",b.fontSize+"px").text(e.get(_).alias??_),{width:E,height:m}=le(c,g,e.get(_).attributes),k=c.insert("rect","#"+u).classed("er entityBox",!0).attr("x",0).attr("y",0).attr("width",E).attr("height",m).node().getBBox();r.setNode(d,{width:k.width,height:k.height,shape:"rect",id:d})}),h},"drawEntities"),he=s(function(t,e){e.nodes().forEach(function(r){r!==void 0&&e.node(r)!==void 0&&t.select("#"+r).attr("transform","translate("+(e.node(r).x-e.node(r).width/2)+","+(e.node(r).y-e.node(r).height/2)+" )")})},"adjustEntities"),bt=s(function(t){return(t.entityA+t.roleA+t.entityB).replace(/\s/g,"")},"getEdgeName"),de=s(function(t,e){return t.forEach(function(r){e.setEdge(X.get(r.entityA),X.get(r.entityB),{relationship:r},bt(r))}),t},"addRelationships"),ut=0,ue=s(function(t,e,r,y,h){ut++;let _=r.edge(X.get(e.entityA),X.get(e.entityB),bt(e)),d=Pt().x(function(w){return w.x}).y(function(w){return w.y}).curve(Yt),c=t.insert("path","#"+y).classed("er relationshipLine",!0).attr("d",d(_.points)).style("stroke",b.stroke).style("fill","none");e.relSpec.relType===h.db.Identification.NON_IDENTIFYING&&c.attr("stroke-dasharray","8,8");let u="";switch(b.arrowMarkerAbsolute&&(u=window.location.protocol+"//"+window.location.host+window.location.pathname+window.location.search,u=u.replace(/\(/g,"\\("),u=u.replace(/\)/g,"\\)")),e.relSpec.cardA){case h.db.Cardinality.ZERO_OR_ONE:c.attr("marker-end","url("+u+"#"+C.ERMarkers.ZERO_OR_ONE_END+")");break;case h.db.Cardinality.ZERO_OR_MORE:c.attr("marker-end","url("+u+"#"+C.ERMarkers.ZERO_OR_MORE_END+")");break;case h.db.Cardinality.ONE_OR_MORE:c.attr("marker-end","url("+u+"#"+C.ERMarkers.ONE_OR_MORE_END+")");break;case h.db.Cardinality.ONLY_ONE:c.attr("marker-end","url("+u+"#"+C.ERMarkers.ONLY_ONE_END+")");break;case h.db.Cardinality.MD_PARENT:c.attr("marker-end","url("+u+"#"+C.ERMarkers.MD_PARENT_END+")");break}switch(e.relSpec.cardB){case h.db.Cardinality.ZERO_OR_ONE:c.attr("marker-start","url("+u+"#"+C.ERMarkers.ZERO_OR_ONE_START+")");break;case h.db.Cardinality.ZERO_OR_MORE:c.attr("marker-start","url("+u+"#"+C.ERMarkers.ZERO_OR_MORE_START+")");break;case h.db.Cardinality.ONE_OR_MORE:c.attr("marker-start","url("+u+"#"+C.ERMarkers.ONE_OR_MORE_START+")");break;case h.db.Cardinality.ONLY_ONE:c.attr("marker-start","url("+u+"#"+C.ERMarkers.ONLY_ONE_START+")");break;case h.db.Cardinality.MD_PARENT:c.attr("marker-start","url("+u+"#"+C.ERMarkers.MD_PARENT_START+")");break}let g=c.node().getTotalLength(),E=c.node().getPointAtLength(g*.5),m="rel"+ut,k=e.roleA.split(/<br ?\/>/g),f=t.append("text").classed("er relationshipLabel",!0).attr("id",m).attr("x",E.x).attr("y",E.y).style("text-anchor","middle").style("dominant-baseline","middle").style("font-family",W().fontFamily).style("font-size",b.fontSize+"px");if(k.length==1)f.text(e.roleA);else{let w=-(k.length-1)*.5;k.forEach((P,Y)=>{f.append("tspan").attr("x",E.x).attr("dy",`${Y===0?w:1}em`).text(P)})}let x=f.node().getBBox();t.insert("rect","#"+m).classed("er relationshipLabelBox",!0).attr("x",E.x-x.width/2).attr("y",E.y-x.height/2).attr("width",x.width).attr("height",x.height)},"drawRelationshipFromLayout"),ye=s(function(t,e,r,y){b=W().er,U.info("Drawing ER diagram");let h=W().securityLevel,_;h==="sandbox"&&(_=at("#i"+e));let d=(h==="sandbox"?at(_.nodes()[0].contentDocument.body):at("body")).select(`[id='${e}']`);C.insertMarkers(d,b);let c;c=new wt({multigraph:!0,directed:!0,compound:!1}).setGraph({rankdir:b.layoutDirection,marginx:20,marginy:20,nodesep:100,edgesep:100,ranksep:100}).setDefaultEdgeLabel(function(){return{}});let u=ce(d,y.db.getEntities(),c),g=de(y.db.getRelationships(),c);Mt(c),he(d,c),g.forEach(function(x){ue(d,x,c,u,y)});let E=b.diagramPadding;Zt.insertTitle(d,"entityTitleText",b.titleTopMargin,y.db.getDiagramTitle());let m=d.node().getBBox(),k=m.width+E*2,f=m.height+E*2;Ft(d,f,k,b.useMaxWidth),d.attr("viewBox",`${m.x-E} ${m.y-E} ${k} ${f}`)},"draw"),pe="28e9f9db-3c8d-5aa5-9faf-44286ae5937c";function Ot(t="",e=""){let r=t.replace(se,"");return`${st(e)}${st(r)}${ne(t,pe)}`}s(Ot,"generateId");function st(t=""){return t.length>0?`${t}-`:""}s(st,"strWithHyphen");var _e={setConf:oe,draw:ye},fe=s(t=>`
  .entityBox {
    fill: ${t.mainBkg};
    stroke: ${t.nodeBorder};
  }

  .attributeBoxOdd {
    fill: ${t.attributeBackgroundColorOdd};
    stroke: ${t.nodeBorder};
  }

  .attributeBoxEven {
    fill:  ${t.attributeBackgroundColorEven};
    stroke: ${t.nodeBorder};
  }

  .relationshipLabelBox {
    fill: ${t.tertiaryColor};
    opacity: 0.7;
    background-color: ${t.tertiaryColor};
      rect {
        opacity: 0.5;
      }
  }

    .relationshipLine {
      stroke: ${t.lineColor};
    }

  .entityTitleText {
    text-anchor: middle;
    font-size: 18px;
    fill: ${t.textColor};
  }    
  #MD_PARENT_START {
    fill: #f5f5f5 !important;
    stroke: ${t.lineColor} !important;
    stroke-width: 1;
  }
  #MD_PARENT_END {
    fill: #f5f5f5 !important;
    stroke: ${t.lineColor} !important;
    stroke-width: 1;
  }
  
`,"getStyles"),Ee=fe,Ne={parser:Wt,db:Gt,renderer:_e,styles:Ee};export{Ne as diagram};
